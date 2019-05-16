####
##### Additional Usefull Layers to KnetLayers
####

## RNN Functionality for handling equal batches faster
struct BiLSTM  <: Layer
    rnn::LSTM
end

function (m::BiLSTM)(x;batchSizes=[1])
    B = first(batchSizes)
    if last(batchSizes) != B
        m.rnn(x;batchSizes=batchSizes,hy=true,cy=false)
    else
        x = reshape(x,size(x,1),B,div(size(x,2),B))
        m.rnn(x;hy=true,cy=false)
    end
end

BiLSTM(input::Int,hidden::Int;o...) =
    BiLSTM(LSTM(input=input, hidden=hidden÷2; bidirectional=true, o...))

## Faster-RCNN needs to be implemented
struct RCNN
    function RCNN(;trained=false)
        new()
    end
end

(::RCNN)(x...) = error("RCNN is not implemented yet")

output_size_of(::Type{RCNN}) = (2048,100)

## CNN for post processing
struct CNN <: Layer
    layers::Chain
end

function (m::CNN)(x; pdrop=0.18)
    for layer in m.layers
        x = layer(dropout(x,pdrop))
    end
    H,W,C,B = size(x)
    return permutedims(reshape(x,H*W,C,B),(2,3,1))
end

CNN(h::Int,w::Int,c::Int,d::Int) = CNN(Chain(Conv(height=h,width=w,inout=c=>d,padding=1,activation=ELU()),
                                             Conv(height=h,width=w,inout=d=>d,padding=1,activation=ELU())))

####
##### Image Processing Units
####

## Image Feature Extractor
struct FeatExtractor{T} <: Layer
    m::T
end

(f::FeatExtractor{Nothing})(x) = x
(f::FeatExtractor)(x) = f.m(x)

function FeatExtractor(;trained=nothing, model=ResNet{101})
    if trained === nothing # do not initialize
        FeatExtractor(nothing)
    else
        if model <: ResNet
            FeatExtractor(model(trained=trained, stage=3))
        else #faster rcnn, needs to be implemented
            FeatExtractor(model(trained=trained))
        end
    end
end

## Image Unit: FeatExtractor + PostProcessor
struct ImageUnit{F, P<:Layer} <: Layer
    feat::FeatExtractor{F} 
    post::P
end

(m::ImageUnit)(x::AbstractString) = m(RGB.(load(x)))
(m::ImageUnit)(x::AbstractMatrix{<:RGB}) = m(m.feat(x))
(m::ImageUnit)(x) = m.post(x)

function ImageUnit(o)
    feat = FeatExtractor(trained=o[:preTrainedFeats], model=o[:featType])
    sizes = output_size_of(o[:featType])
    if o[:featPostProcess] <: CNN
        ImageUnit(feat, CNN(3,3,sizes[3],o[:d]))
    elseif o[:featPostProcess] <: Dense || o[:featPostProcess] <: Linear
        ImageUnit(feat, o[:featPostProcess](input=sizes[1], output=o[:d])) # Linear Layers
    else
        error("Image Unit specifications cannot be satisfied!")
    end
end

####
##### Question Processing Unit
####
struct QUnit  <: Layer
    embed::Embed
    rnn::BiLSTM
end

function (m::QUnit)(x; batchSizes::Vector{Int}=[1], edrop::Real=0.2, qdrop::Real=0.08)
    xe  = dropout(m.embed(x), edrop)
    out = m.rnn(xe; batchSizes=batchSizes)
    q   = dropout(vcat(out.hidden[:,:,1], out.hidden[:,:,2]), qdrop)
    if last(batchSizes) != first(batchSizes) #padding
        cws = PadRNNOutput(out.y,_batchSizes2indices(batchSizes)) #add zero padding to rnn output
        return q,cws
    end
    return q, out.y
end

QUnit(o, embedding::Integer) = 
    QUnit(o, Embed(input=o[:qVocabSize], output=o[:embedSize]))

QUnit(o, embedding::AbstractArray) = 
    QUnit(o, Embed(param(convert(arrtype,copy(embedding)))))

QUnit(o, embedding::Embed) =
    QUnit(embedding,BiLSTM(o[:embedSize],o[:d]))

###
#### MAC Units
###

## Control Unit
struct Control  <: Layer
    cq::Union{Linear,Nothing}
    att::Linear
end

function (m::Control)(c, q, cws, mask; tap=nothing)
    d,B,T = size(cws)
    cqi = m.cq === nothing ? q : m.cq(vcat(c,q))
    cvis  = reshape(cqi .* cws,(d,B*T))
    cvis_2d = reshape(m.att(cvis),(B,T)) #eq c2.1.2
    if mask === nothing
        cvi = reshape(softmax(cvis_2d, dims=2),(1,B,T)) #eq c2.2
    else
        cvi = reshape(softmax(cvis_2d .+ mask,dims=2),(1,B,T)) #eq c2.2
    end
    tap===nothing || get!(tap,"w_attn_$(tap["cnt"])",Array(reshape(cvi,B,T)))
    cnew = reshape(sum(cvi.*cws;dims=3),(d,B))
end

function Control(d::Int; prevControl=false)
    if prevControl
        Control(Linear(input=2d,output=d),Linear(input=d,output=1))
    else
        Control(nothing,Linear(input=d,output=1))
    end
end

##  Read Unit
struct Read  <: Layer
    me::Linear
    Kbe::Linear
    Kbe2::Linear
    Ime
    att::Linear
end

function (m::Read)(mp,ci,cws,KBhw,mask; tap=nothing)
    d,B,N = size(KBhw); BN = B*N
    mp    = dropout(mp,0.15)
    mproj = m.me(dropout(mp,0.15))
    KBhw′ = m.Kbe(dropout(KBhw,0.15))
    ImKB  = reshape(mproj .* KBhw′,(d,BN)) # eq r1.2
    ImKB′ = reshape(elu.(m.Ime*ImKB .+ m.Kbe2(reshape(KBhw′,(d,BN)))),(d,B,N)) #eq r2
    IcmKB_pre = elu.(reshape(ci .* ImKB′,(d,BN))) #eq r3.1.1
    IcmKB_pre = dropout(IcmKB_pre,0.15)
    IcmKB = reshape(m.att(IcmKB_pre),(B,N)) #eq r3.1.2
    if mask===nothing
        mvi = reshape(softmax(IcmKB,dims=2),(1,B,N)) #eq r3.2
    else
        mvi = reshape(softmax(IcmKB .+ mask,dims=2),(1,B,N)) #eq r3.2
    end
    tap===nothing || get!(tap,"KB_attn_$(tap["cnt"])",Array(reshape(mvi,B,N)))
    mnew = reshape(sum(mvi.*KBhw;dims=3),(d,B)) #eq r3.3
end

Read(d::Int) = Read(Linear(input=d,output=d),Linear(input=d,output=d),
                    Linear(input=d,output=d),param(d,d; atype=arrtype, init=xavier),
                    Linear(input=d,output=1))

## Write Unit
struct Write  <: Layer
    me::Linear
    cproj::Union{Linear,Nothing}
    att::Union{Linear,Nothing}
    mpp
    gating::Union{Linear,Nothing}
end

function (m::Write)(m_new,mi₋1,mj,ci,cj; tap=nothing)
    d,B        = size(m_new)
    mi         = m.me(vcat(m_new,mi₋1))
    m.att===nothing && return mi
    T          = length(mj)
    ciproj     = m.cproj(ci)
    cj_3d      = reshape(cat1d(cj...),(d,B,T)) #reshape(hcat(cj...),(d,B,T)) #
    sap        = reshape(ciproj.*cj_3d,(d,B*T)) #eq w2.1.1
    sa         = reshape(m.att(sap),(B,T)) #eq w2.1.2
    sa′        = reshape(softmax(sa,dims=2),(1,B,T)) #eq w2.1.3
    mj_3d      = reshape(cat1d(mj...),(d,B,T)) #reshape(hcat(mj...),(d,B,T)) #
    mi_sa      = reshape(sum(sa′ .* mj_3d;dims=3),(d,B))
    mi′′       = m.mpp*mi_sa .+ mi #eq w2.3
    m.gating===nothing && return mi′′
    σci′       = sigm.(m.gating(ci))  #eq w3.1
    mi′′′      = (σci′ .* mi₋1) .+  ((1 .- σci′) .* mi′′) #eq w3.2
end

function Write(d::Int; selfattn=true, gating=true)
    if selfattn
        if gating
            Write(Linear(input=2d,output=d),Linear(input=d,output=d),
                  Linear(input=d,output=1),param(d,d;atype=arrtype, init=xavier),
                  Linear(input=d,output=1))
        else
            Write(Linear(input=2d,output=d),Linear(input=d,output=d),
                  Linear(input=d,output=1),param(d,d;atype=arrtype, init=xavier),
                  nothing)
        end
    else
        Write(Linear(input=2d,output=d),nothing,nothing,nothing,nothing)
    end
end

struct MAC <: Layer
    control::Control
    read::Read
    write::Write
end

function (m::MAC)(qi,cws,mi,mj,ci,cj,KBhw,cw_mask,kb_mask; tap=nothing)
    cnew = m.control(ci,qi,cws,cw_mask; tap=tap)
    ri   = m.read(mi,cnew,cws,KBhw,kb_mask; tap=tap)
    mnew = m.write(ri,mi,mj,cnew,cj)
    return cnew,mnew
end

MAC(o) = MAC(Control(o[:d]; prevControl=o[:prevControl]),
             Read(o[:d]),
             Write(o[:d]; selfattn=o[:selfattn],gating=o[:gating]))

###
#### Output Unit
###

struct Output <: Layer
    qe::Linear
    l1::Dense
    l2::Linear
end

function (m::Output)(q,mp; pdrop::Real=0.15)
    qe  = m.qe(q)
    x1  = m.l1(dropout(cat(qe,mp,mp.*qe;dims=1), pdrop))
    return m.l2(dropout(x1, pdrop))
end

Output(o) = Output(Linear(input=o[:d],output=o[:d]),
                   Dense(input=3o[:d],output=o[:d],activation=ELU()),
                   Linear(input=o[:d],output=o[:aVocabSize]))

## Prediction Functions

function make_predictions(y; unk::Int=2)
    predmat = convert(Array{Float32},y)
    predmat[unk,:] .-= 1.0f30 #not predict unk:2
    predictions = mapslices(argmax,predmat,dims=1)[1,:]
end

make_all_predictions(M, q, m_history, p) =
    [make_predictions(M.output(q,m_history[i])) for i=1:p-1]


####
##### MAC Network
####
struct MACNetwork <: Layer
    imgunit::ImageUnit
    qunit::QUnit
    qindex::Linear
    mac::MAC
    output::Output
    m0
    loss::SigmoidCrossEntropyLoss
end

function (M::MACNetwork)(questions, batchSizes, feats,
                         cw_mask=nothing,
                         kb_mask=nothing;
                         p=4,
                         answers=nothing,
                         tap=nothing,
                         allsteps=false)
    
    train   = answers !== nothing
    # Feature Extraction
    #feats = M.imgunit.feat(feats)
    
    # Feature Post Processing
    # FIXME: L2 Norm should be inside of Image Unit!!!
    KBhw = M.imgunit(l2_normalize(feats,dims=1)) #knowledge base

    d,B,N  = size(KBhw) 

    # Question Unit
    q,cws  = M.qunit(questions; batchSizes=batchSizes)
    qis    = M.qindex(q)
    
    # Memory Initialization
    ci,mi,c_history,m_history = init_state(M,q)
    
    # BODY: MAC Cells
    for i=1:p
        qi = qis[(i-1)*d+1:i*d,:]
        ci, mi = M.mac(qi,cws,mi,m_history,ci,c_history,KBhw,cw_mask,kb_mask;tap=tap)
        if M.mac.write.att !== nothing # self-attention
            push!(c_history,ci);
            push!(m_history,mi);
        end
        tap!==nothing && (tap["cnt"]+=1)
    end
    
    # Output Logits
    y = M.output(q,mi)

    tap!==nothing && (tap["y"]=value(y))
    if !train
        if allsteps && m_history !== nothing
            make_all_predictions(M, q, m_history, p)
        else
            make_predictions(y)
        end
    else
        z = one_hot(size(y), answers; atype=arrtype)
        return M.loss(y,z)
    end
end

function MACNetwork(o::Dict;embeddings=o[:embedSize])
           MACNetwork(ImageUnit(o),
                      QUnit(o,embeddings),
                      Linear(input=o[:d],output=o[:p]*o[:d]),
                      MAC(o),
                      Output(o),
                      param(o[:d],1;atype=arrtype, init=randn),
                      SigmoidCrossEntropyLoss(dims=1)
                      )
end

function init_state(M,q)
    mi = M.m0*ones(arrtype,1,size(q,2))
    if M.mac.write.att !== nothing
        cj=[q]; mj=[mi]
    else
        cj=nothing; mj=nothing
    end
    return q,mi,cj,mj
end

setoptim!(m::MACNetwork,o) = 
    for param in params(m); param.opt = Adam(;lr=o[:lr]); end

lrdecay!(M::MACNetwork, decay::Real) =
    for p in params(M); p.opt.lr = p.opt.lr*decay; end

function loadImageBatch(feats, indices::Vector{Int}, ::Type{<:RCNN})
    feats_L, feats_H, feats_C = 204800, 2048, 100
    B = length(indices)
    totlen = feats_L*B
    result = Array{Float32}(undef, totlen)
    starts = (0:B-1) .*feats_L .+ 1; ends = starts .+ feats_L .- 1;
    for i=1:B
        result[starts[i]:ends[i]] = view(feats,:,:,indices[i])
    end
    return permutedims(reshape(result,feats_H,feats_C,B),(1,3,2))
end

function loadImageBatch(feats, indices::Vector{Int}, ::Type{<:ResNet})
    feats_L, feats_H, feats_C = 200704, 14, 1024
    B = length(args)
    totlen = feats_L*B
    result = similar(Array{Float32}, totlen)
    starts = (0:B-1) .*feats_L .+ 1; ends = starts .+ feats_L .- 1;
    for i=1:B
        result[starts[i]:ends[i]] = view(feats,:,:,:,args[i])
    end
    return reshape(result,feats_H,feats_H,feats_C,B)
end

####
##### KnetLayers Additional Functionalities
####
@inline Base.zeros(T::Type{<:KnetArray}, d::Int...) = fill!(T(undef,d...),zero(eltype(T)))
@inline Base.ones(T::Type{<:KnetArray}, d::Int...) = fill!(T(undef,d...),one(eltype(T)))
output_size_of(::Type{<:ResNet})  = (14,14,1024)

function PadRNNOutput(y, indices)
    d      = size(y,1)
    B      = length(indices)
    lngths = length.(indices)
    Tmax   = first(lngths)
    cw = []
    @inbounds for i=1:B
        y1 = y[:,indices[i]]
        df = Tmax-lngths[i]
        if df > 0
            pad  = zeros(arrtype, d*df)
            ypad = reshape(cat1d(y1,pad),d,Tmax) # hcat(y1,kpad)
            push!(cw,ypad)
        else
            push!(cw,y1)
        end
    end
    return reshape(vcat(cw...),d,B,Tmax)
end
## FIXME: Similar to maskQuestions, so merge it
function maskQuestions(lengths::Vector{Int})
    Tmax = first(lengths)
    Tmax == last(lengths) && return nothing
    B    = length(lengths)
    mask = falses(length(lengths), Tmax)
    for k=1:B
        @inbounds mask[k,lengths[k]+1:Tmax] .= true
    end
    return mask
end

function maskObjects(objectnums, Omax::Int=100)
    B = length(objectnums)
    mask = falses(length(objectnums),Omax)
    for k=1:B
        @inbounds mask[k,objectnums[k]+1:Omax] .= true
    end
    return mask
end

l2_normalize(x; dims=:) = x ./ sqrt.(max.(Knet.sumabs2(x, dims=dims),1e-12))

#FIXME: Benchmarks
function benchmark(M::MACNetwork,feats,o;N=10)
    getter(id) = view(feats,:,:,:,id)
    B=32;L=25
    @time for i=1:N
        ids  = randperm(128)[1:B]
        xB   = arrtype(ones(Float32,1,B))
        xS   = arrtype(batcher(map(getter,ids)))
        xQ   = [rand(1:84) for i=1:B*L]
        answers = [rand(1:28) for i=1:B]
        batchSizes = [B for i=1:L]
        xP   = nothing
        y    = @diff M(xQ,batchSizes,xS,xB,xP;answers=answers,p=o[:p])
    end
end

function benchmark(feats,o;N=30)
    M     = MACNetwork(o);
    benchmark(M,feats,o;N=N)
end


