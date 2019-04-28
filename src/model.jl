import KnetLayers: arrtype, Activation, Filtering

abstract type Model end
struct ResNet <: Model; w; end
function (M::ResNet)(m,imgurl::String,avgimg;stage=3)
    img = imgdata(imgurl, avgimg)
    return M(w,m,arrtype(img);stage=stage);
end
function ResNet(atype::Type;stage=3)
    w,m,meta = ResNetLib.resnet101init(;trained=true,stage=stage)
    global avgimg = meta["normalization"]["averageImage"]
    global descriptions = meta["classes"]["description"]
    return w,m,meta,avgimg
end

ResNet() = ResNet(nothing);

struct CNN <: Model
    layer1::Filtering{typeof(conv4)}
    layer2::Filtering{typeof(conv4)}
    drop::Dropout
end
function (m::CNN)(x)
    x1 = m.layer1(m.drop(x))
    x2 = m.layer2(m.drop(x1))
    h,w,c,b = size(x2)
    permutedims(reshape(x2,h*w,c,b),(2,3,1))
end
CNN(h::Int,w::Int,c::Int,d::Int) = CNN(Conv(height=h,width=w,inout=c=>d,padding=1,activation=ELU()),
                                       Conv(height=h,width=w,inout=d=>d,padding=1,activation=ELU()),
                                       Dropout(0.18))

# struct Stem <: Model
#     layer::Linear
# end
# Stem(input=2048,output=512) = Stem(Linear(input=input,output=output))
# (m::Stem)(x) = m.layer(x) # should be 512xBX100 

struct mRNN  <: Model
    rnn::LSTM
end

function (m::mRNN)(x;batchSizes=[1])
    B = first(batchSizes)
    if last(batchSizes)!=B
        out = m.rnn(x;batchSizes=batchSizes,hy=true,cy=false)
    else
        x   = reshape(x,size(x,1),B,div(size(x,2),B))
        out = m.rnn(x;hy=true,cy=false)
    end
    return out.y, out.hidden
end
mRNN(input::Int,hidden::Int;o...) = mRNN(LSTM(input=input, hidden=hidden÷2;o...))
mRNN(input,hidden::Int;o...) = mRNN(LSTM(input=size(input,1), hidden=hidden÷2;o...))


struct QUnit  <: Model
    embed::Embed
    rnn::mRNN
    #linear::Linear
    drop1::Dropout
    drop2::Dropout
end
function (m::QUnit)(x;batchSizes=[1],train=false)
    xe = m.drop1(m.embed(x))
    y,hyout = m.rnn(xe;batchSizes=batchSizes)
    q  = m.drop2(vcat(hyout[:,:,1],hyout[:,:,2]))
    B = batchSizes[1]
    
    if ndims(y) == 2
        indices      = bs2ind(batchSizes)
        lngths       = length.(indices)
        Tmax         = maximum(lngths)
        td,B         = size(q)
        d            = div(td,2)
        cw           = Any[];
        for i=1:length(indices)
            y1 = y[:,indices[i]]
            df = Tmax-lngths[i]
            if df > 0
                cpad = zeros(Float32,2d*df) # zeros(Float32,2d,df)
                kpad = arrtype(cpad)
                ypad = reshape(cat1d(y1,kpad),2d,Tmax) # hcat(y1,kpad)
                push!(cw,ypad)
            else
                push!(cw,y1)
            end
        end
        cws_2d =  reshape(vcat(cw...),2d,B*Tmax)
    else
        d      = div(size(y,1),2)
        Tmax   = size(y,3)
        cws_2d = reshape(y,2d,B*Tmax)
    end
    cws_3d =  reshape(cws_2d,(2d,B,Tmax))
    return q,cws_3d;
end

KnetLayers.Embed(vocab::Int,embed::Int)   = Embed(input=vocab, output=embed; winit=rand)
KnetLayers.Embed(vocab::Int,embed)        = Embed(embed)
Base.size(l::KnetLayers.Multiply,x...) = size(l.weight,x...)

QUnit(vocab::Int,embed,hidden::Int;bidir=true) = QUnit(Embed(vocab,embed),
                                                       mRNN(embed,hidden;bidirectional=bidir),
                                                       #Linear(input=2hidden,output=hidden),
                                                       Dropout(0.2), Dropout(0.08))

function bs2ind(batchSizes)
    B = batchSizes[1]
    indices = Any[]
    for i=1:B
        ind = i.+cumsum(filter(x->(x>=i),batchSizes)[1:end-1])
        push!(indices,append!(Int[i],ind))
    end
    return indices
end

struct Control  <: Model
   # cq::Linear
    att::Linear
end
function (m::Control)(c,q,cws,pad;train=false,tap=nothing)
      d,B,T = size(cws)
      cqi   = q # reshape(m.cq(vcat(c,q)),(d,B,1))
      cvis  = reshape(cqi .* cws,(d,B*T))
      cvis_2d = reshape(m.att(cvis),(B,T)) #eq c2.1.2
      if pad != nothing
          cvi = reshape(softmax(cvis_2d .- pad,dims=2),(1,B,T)) #eq c2.2
      else
          cvi = reshape(softmax(cvis_2d,dims=2),(1,B,T)) #eq c2.2
      end
      tap!=nothing && get!(tap,"w_attn_$(tap["cnt"])",Array(reshape(cvi,B,T)))
      cnew = reshape(sum(cvi.*cws;dims=3),(d,B))
end
Control(d::Int) = Control(Linear(input=d,output=1)) #cq::Linear(input=2d,output=d)

struct Read  <: Model
    me::Linear
    Kbe::Linear
    Kbe2::Linear
    Ime
    att::Linear
    drop::Dropout
end

function (m::Read)(mp,ci,cws,KBhw, pad;train=false,tap=nothing)
    d,B,N = size(KBhw); BN = B*N
    mp    = dropout(mp,0.15)
    mi_3d = reshape(m.me(dropout(mp,0.15)),(d,B,1))
    KBhw′ = m.Kbe(dropout(KBhw,0.15))
    ImKB  = reshape(mi_3d .* KBhw′,(d,BN)) # eq r1.2
    ImKB′ = reshape(elu.(m.Ime*ImKB .+ m.Kbe2(reshape(KBhw′,(d,BN)))),(d,B,N)) #eq r2
    ci_3d = reshape(ci,(d,B,1))
    IcmKB_pre = elu.(reshape(ci_3d .* ImKB′,(d,BN))) #eq r3.1.1
    IcmKB_pre = m.drop(IcmKB_pre)
    IcmKB = reshape(m.att(IcmKB_pre),(B,N)) #eq r3.1.2
    mvi = reshape(softmax(IcmKB .- pad,dims=2),(1,B,N)) #eq r3.2
    tap!=nothing && get!(tap,"KB_attn_$(tap["cnt"])",Array(reshape(mvi,B,N)))
    mnew = reshape(sum(mvi.*KBhw;dims=3),(d,B)) #eq r3.3
end
Read(d::Int) = Read(Linear(input=d,output=d),Linear(input=d,output=d),
                    Linear(input=d,output=d),param(d,d; atype=arrtype, init=xavier),
                    Linear(input=d,output=1), Dropout(0.15))

struct Write  <: Model
    me::Linear
    cproj::Union{Linear,Nothing}
    att::Union{Linear,Nothing}
    mpp
    gating::Union{Linear,Nothing}
end

function (m::Write)(m_new,mi₋1,mj,ci,cj;train=false,selfattn=true,gating=true,tap=nothing)
    d,B        = size(m_new)
    mi         = m.me(vcat(m_new,mi₋1))
    !selfattn && return mi
    T          = length(mj)
    ciproj     = m.cproj(ci)
    ci_3d      = reshape(ciproj,d,B,1)
    cj_3d      = reshape(cat1d(cj...),(d,B,T)) #reshape(hcat(cj...),(d,B,T)) #
    sap        = reshape(ci_3d.*cj_3d,(d,B*T)) #eq w2.1.1
    sa         = reshape(m.att(sap),(B,T)) #eq w2.1.2
    sa′        = reshape(softmax(sa,dims=2),(1,B,T)) #eq w2.1.3
    mj_3d      = reshape(cat1d(mj...),(d,B,T)) #reshape(hcat(mj...),(d,B,T)) #
    mi_sa      = reshape(sum(sa′ .* mj_3d;dims=3),(d,B))
    mi′′       = m.mpp*mi_sa .+ mi #eq w2.3
    !gating && return mi′′
    σci′       = sigm.(m.gating(ci))  #eq w3.1
    mi′′′      = (σci′ .* mi₋1) .+  ((1 .- σci′) .* mi′′) #eq w3.2
end

function Write(d::Int;selfattn=true,gating=true)
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

struct MAC <: Model
    control::Control
    read::Read
    write::Write
end
function (m::MAC)(qi,cws,mi,mj,ci,cj,KBhw,pad,opad;train=false,selfattn=true,gating=true,tap=nothing)
    cnew = m.control(ci,qi,cws,pad;train=train,tap=tap)
    ri   = m.read(mi,cnew,cws,KBhw,opad;train=train,tap=tap)
    mnew = m.write(ri,mi,mj,cnew,cj;train=train,selfattn=selfattn,gating=gating)
    return cnew,mnew
end
MAC(d::Int;selfattn=false,gating=false) = MAC(Control(d),Read(d),Write(d;selfattn=selfattn,gating=gating))

struct Output <: Model
    qe::Linear
    l1::Dense
    l2::Linear
end

function (m::Output)(q,mp)
    eq = m.qe(q)
    x  = dropout(cat(eq,mp,mp.*eq;dims=1),0.15)
    return m.l2(dropout(m.l1(x),0.15))
end


Output(d::Int) = Output(Linear(input=d,output=d),
                        Dense(input=3d,output=d,activation=ELU()),
                        Linear(input=d,output=1845))

struct MACNetwork <: Model
    resnet::ResNet
    stem::Linear
    qunit::QUnit
    qindex::Linear
    mac::MAC
    output::Output
    #c0
    m0
    drop::Dropout
end

l2_normalize(x;dims=:) = x ./ sqrt.(max.(Knet.sumabs2(x, dims=dims),1e-12))

function (M::MACNetwork)(qs,batchSizes,xS,xB,xP, opad;answers=nothing,p=12,selfattn=false,gating=false,tap=nothing,allsteps=false)
    train         = answers!=nothing
    #STEM Processing
    KBhw          = M.stem(l2_normalize(xS,dims=1))
    #Read Unit Precalculations
    d,B,N         = size(KBhw)
    #KBhw_2d       = M.drop(reshape(KBhw,(dQ,B*N)))
    #KBhw′_pre     = M.mac.read.Kbe(KBhw_2d) # look if it is necessary
    #KBhw′′        = M.mac.read.Kbe2(KBhw′_pre)
    #KBhw′         = reshape(KBhw′_pre,(d,B,N))

    #Question Unit
    q,cws         = M.qunit(qs;batchSizes=batchSizes,train=train)
    qi_c          = M.qindex(q)
    #Memory Initialization
    ci            = q#M.c0*xB
    mi            = M.m0*xB

    if selfattn
        cj=[ci]; mj=[mi]
    else
        cj=nothing; mj=nothing
    end

    for i=1:p
        qi = qi_c[(i-1)*d+1:i*d,:]
        #ci = M.drop(ci)
        ci,mi = M.mac(qi,cws,mi,mj,ci,cj,KBhw,xP,opad;train=train,selfattn=selfattn,gating=gating,tap=tap)
	#mi = M.drop(mi)
        if selfattn; push!(cj,ci); push!(mj,mi); end
        tap!=nothing && (tap["cnt"]+=1)
    end

    y = M.output(q,mi)    

    if answers==nothing
        predmat = convert(Array{Float32},y)
        predmat[2,:] .-= 1.0f30 #not predict unk:2
        tap!=nothing && get!(tap,"y",predmat)
        predictions = mapslices(argmax,predmat,dims=1)[1,:]
        if allsteps
            outputs = []
            for i=1:p-1
                yi = M.output(q,mj[i])
                yi = convert(Array{Float32},yi)
                push!(outputs,mapslices(argmax,yi,dims=1)[1,:])
            end
            push!(outputs,predictions)
            return outputs
        end
        return predictions
    else
        return nll(y,answers)
    end
end

function MACNetwork(o::Dict;embed=o[:embed_size])
           MACNetwork(ResNet(),
                      Linear(input=2048,output=o[:d]),
                      QUnit(o[:vocab_size],embed,o[:d]),
                      Linear(input=o[:d],output=o[:p]*o[:d]),
                      MAC(o[:d];selfattn=o[:selfattn],gating=o[:gating]),
                      Output(o[:d]),
                      #param(o[:d],2o[:d];atype=arrtype, init=xavier),
                      param(o[:d],1;atype=arrtype, init=randn),
                      Dropout(0.15))
end

setoptim!(m::MACNetwork,o) = 
    for param in params(m); param.opt = Adam(;lr=o[:lr]); end

lrdecay!(M::MACNetwork, decay::Real) =
    for p in params(M); p.opt.lr = p.opt.lr*decay; end

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
        y    = @diff M(xQ,batchSizes,xS,xB,xP;answers=answers,p=o[:p],selfattn=o[:selfattn],gating=o[:gating])
    end
end
function benchmark(feats,o;N=30)
    M     = MACNetwork(o);
    benchmark(M,feats,o;N=N)
end


const feats_L = 204800;
const feats_H = 2048;
const feats_C = 100;
function batcher1(feats,args)
    B = length(args)
    totlen = feats_L*B
    result = Array{Float32}(undef, totlen)
    starts = (0:B-1) .*feats_L .+ 1; ends = starts .+ feats_L .- 1;
    for i=1:B
        result[starts[i]:ends[i]] = view(feats,:,:,args[i])
    end
    return permutedims(reshape(result,feats_H,feats_C,B),(1,3,2))
end

function inplace_batcher(result,data,args)
     B = length(args)
     totlen = feats_L*B
     starts = (0:B-1) .* feats_L .+ 1; ends = starts .+ feats_L .- 1;
     for i=1:B       
          result[starts[i]:ends[i]] = view(data,:,:,:,args[i])
     end
     return reshape(result,feats_H,feats_H,feats_C,B)
end

sigm_loss(x,z) = relu.(x) .- x .* z - log.(sigm.(abs.(x)))

function sloss(y,a::AbstractArray{<:Integer})
    indices = Knet.findindices(y,a,dims=1)
    labels = zeros(Float32,size(y))
    labels[indices] .= 1.0f0
    z = arrtype(labels)
    lp = sum(sigm_loss(y,z),dims=:)
    return lp/length(a)
end
