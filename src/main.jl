using ImageMagick, Images
using AutoGrad, Knet, KnetLayers, JSON, HDF5
using Printf, Random
import KnetLayers: IndexedDict, arrtype, Activation, Filtering, Layer,
                  _batchSizes2indices, PadRNNOutput, one_hot, _pack_sequence
include(KnetLayers.dir("examples/resnet.jl")) #load resnet functionalities
using .ResNetLib

include("model.jl")

savemodel(filename,m,mrun,o) =
    Knet.save(filename,"m",m,"mrun",mrun,"o",o)

function loadmodel(filename;onlywrun=false)
    d = Knet.load(filename)
    if onlywrun
        mrun=d["mrun"];o=d["o"]
        m=nothing
    else
        m=d["m"];o=d["o"];mrun=d["mrun"];
    end
    return m,mrun,o;
end

function getQdata(dhome,set; fname="objects")
    JSON.parsefile(dhome*"gen_gqa_"*fname*"_"*set*"Instances.json")
end

function getDicts(dhome,dicfile;fname="objects")
    dic  = JSON.parsefile(dhome*"vocab.json")
    words  = IndexedDict(convert(Dict{String,Int},dic["word_dic"]))
    answer = IndexedDict(convert(Dict{String,Int},dic["answer_dic"]))
    id2index = JSON.parsefile(dhome*"gqa_$(fname)_merged_info.json")
    return words,answer,id2index
end

function loadFeatures(dhome;h5=false, featsize=(2048,100), fname="objects")
    if h5
        h5open(dhome*"gqa_"*fname*".h5","r")["features"]
    else
        feats = reinterpret(Float32,read(open(dhome*"all_"*fname*".bin")))
        reshape(feats,(featsize...,div(length(feats),prod(featsize))))
    end
end

unzip(d; range=fieldnames(eltype(d))) = map(x->getfield.(d, x),range)
function miniBatch(data, q2i, a2i, id2index; shfl=true, srtd=false, B=64)
    L = length(data)
    shfl && shuffle!(data)
    srtd && sort!(data;by=x->length(x["question"]))
    batchs = [];
    for i=1:B:L
        batch = map(1:min(L-i+1,B)) do j
             d = data[i+j-1]
            (map(x->get(q2i,x,"<UNK>"),d["question"]), 
             Int(id2index[d["imageId"]["id"]]["index"]),
             get(a2i,d["answer"],"<UNK>"),
             1, # FIXME: questionFamily
             Int(get(d,"objectsNum",100)))
        end
        batch = sort!(batch, by=d->length(d[1]), rev=true)
        quesvecs, images, answers, families, objectnums = unzip(batch; range=1:5)
        questions, batchSizes = _pack_sequence(quesvecs)
        q_mask  = maskQuestions(length.(quesvecs))
        kb_mask = maskObjects(objectnums)
        push!(batchs,(images, questions, batchSizes, answers, families, objectnums, q_mask, kb_mask))
    end
    return batchs
end

function loadTrainingData(dhome="data/";h5=false)
    !h5 && println("Loading pretrained features for train&val sets.
                It requires minimum 95GB RAM!!!")
    feats = loadFeatures(dhome;h5=h5)
    println("Loading questions ...")
    trnqstns = getQdata(dhome,"train")
    valqstns = getQdata(dhome,"val")
    println("Loading dictionaries ... ")
    _,_,id2index = getDicts(dhome,"dic")
    d = load(dhome*"wordembeddings_vocabs.jld2")
    qvoc,avoc,embeddings = d["word_dict"],d["answer_dict"],d["embeddings"]
    return feats,(trnqstns,valqstns),(qvoc,avoc,id2index),embeddings
end

function loadDemoData(dhome="data/demo/",featsize=(2048,100))
    println("Loading demo features ...")
    feats = reinterpret(Float32,read(open(dhome*"demo.bin")))
    feats = reshape(feats,(featsize...,div(length(feats),prod(featsize))))
    println("Loading demo questions ...")
    qstns = JSON.parsefile(dhome*"demo.json")
    println("Loading dictionaries ...")
    d = KnetLayers.load(dhome*"dict.jld2")
    dics = (d["word_dict"],d["answer_dict"])
    return feats,qstns,dics
end

exp_mask(mask, atype) = atype(mask*(-1.0f22))
exp_mask(mask::Nothing, atype) = nothing

function modelrun(M,data,feats,o,Mrun=nothing; train::Bool=false, interval::Int=1000)
    ft = eltype(arrtype)
    cnt, total, L = ft(0), ft(0), length(data)
    Mp, MRp  = params(M), params(Mrun)
    println(@sprintf("%.2f Accuracy|Loss", train ? cnt/total : 100cnt/total))
    for (t,i) in progress(enumerate(randperm(L)))
        images, questions, batchSizes, answers,_,_,q_mask,kb_mask = data[i]
        total += B = length(images)
        ximg  = arrtype(loadImageBatch(feats,images,o[:featType]))
        qmask, kbmask = exp_mask(q_mask, arrtype), exp_mask(kb_mask, arrtype)
        if train
            J = @diff M(questions,batchSizes,ximg,qmask,kbmask; answers=answers, p=o[:p])
            update_with_gclip(J, Mp; clip=ft(8))
            cnt += value(J)*B
            Mrun===nothing || ema_apply!(Mp, MRp, ft(o[:ema]))
        else
            preds = M(questions,batchSizes,ximg,qmask,kbmask; p=o[:p])
            cnt += sum(preds .== answers)
        end
        t%interval==0 && println(@sprintf("%.2f Accuracy|Loss", train ? cnt/total : 100cnt/total))
    end
    println(@sprintf("%.2f Accuracy|Loss", train ? cnt/total : 100cnt/total))
    train && savemodel(o[:prefix]*".jld2",M,Mrun,o);
    return cnt/total
end

function ema_apply!(Mparams, Rparams, ema::Real)
    for (wr,wi) in zip(Rparams,Mparams);
        Knet.axpy!(1-ema,wi.value-wr.value,wr.value);
    end
end

function Base.copyto!(Mdest,Msource)
    for (wr,wi) in zip(params(Mdest),params(Msource));
        value(wr)[:] = value(wi)[:] 
    end
    return Mdest
end


function train!(M,Mrun,sets,feats,o)
    @info "Training Starts...."
    setoptim!(M,o)
    minloss = typemax(Float32)
    for i=1:o[:epochs]
        println("Epoch $(i) starts...")
        trnloss = modelrun(M,sets[1],feats,o,Mrun;train=true)
        if trnloss > minloss
            lrdecay!(M,0.5f0)
        else
            minloss = trnloss
        end
        for k=2:length(sets)
            modelrun(Mrun,sets[k],feats,o;train=false)
        end
    end
    return M,Mrun;
end

function train(sets,feats,o;embed=300)
     if o[:mfile]==nothing
         M    = MACNetwork(o;embed=embed)
         Mrun = deepcopy(M)
     else
         M,Mrun,o = loadmodel(o[:mfile])
     end
     train!(M,Mrun,sets,feats,o)
     return M,Mrun;
end

function train(dhome="data/",o=nothing)
     if o==nothing
         o=Dict(:h5=>false,:mfile=>nothing,:epochs=>25,
                :lr=>0.0001,:p=>4,:ema=>0.999f0,:batchsize=>64,
                :selfattn=>false,:gating=>false,:d=>512,
                :shuffle=>true,:sorted=>false,:prefix=>string(now())[1:10],
                :vocab_size=>2960,:embed_size=>300, :dhome=>"data/", :loadresnet=>false)
     end
     feats,qdata,dics,embeddings = loadTrainingData(dhome;h5=o[:h5])
     sets = []
     for q in qdata; push!(sets,miniBatch(q,dics...;shfl=o[:shuffle],srtd=o[:sorted])); end
     qdata = nothing; #gc();
     M,Mrun = train(sets,feats,o;embed=Param(arrtype(embeddings)))
     return M,Mrun,sets,feats,dics;
end

function validate(Mrun,valset,feats,o)
     modelrun(Mrun,valset,feats,o;train=false)
end

function validate(mfile,valset,feats,o)
     _,Mrun,_ = loadmodel(mfile)
     modelrun(Mrun,valset,feats;train=false)
     return Mrun
end

function validate(mfile,dhome,o)
     _,Mrun,_,o   = loadmodel(mfile)
     feats        = loadFeatures(dhome)
     qdata        = getQdata(dhome,"val")
     dics         = getDicts(dhome,"dic")
     valset       = miniBatch(qdata,dics...;shfl=o[:shuffle],srtd=o[:sorted])
     modelrun(Mrun,valset,feats,o;train=false)
     return Mrun,valset,feats
end

function singlerun(Mrun,ximg,question;p=4,objectnum=100)
    results    = Dict{String,Any}("cnt"=>1)
    batchSizes = ones(Int,length(question))
    kb_mask    = exp_mask(maskObjects([objectnum]),arrtype)
    outputs = Mrun(question,batchSizes,ximg,nothing,kb_mask;tap=results,p=p,allsteps=true)
    prediction = make_predictions(results["y"])
    return results,prediction,outputs
end

function visualize(img,results;p=12)
    s_y,s_x = size(img)./14
    for k=1:p
        α = results["w_attn_$(k)"][:]
        wrds    = i2w[question]
        p = bar(α;xticks=(collect(1:length(wrds)),wrds),xrotation=90,bar_width = 1,
            xtickfont = font(8, "Courier"),yticks=0:.1:(maximum(α)+.1),
            legend=false,size=(600,100+400*(maximum(α))),aspect_ratio=10)      
        savefig(p,"plots/$(k).png")
        println("Image Attention Map $k: ")
        flush(stdout)
        hsvimg = HSV.(img);
        attn = results["KB_attn_$(k)"]
        for i=1:14,j=1:14
            rngy          = floor(Int,(i-1)*s_y+1):floor(Int,min(i*s_y,320))
            rngx          = floor(Int,(j-1)*s_x+1):floor(Int,min(j*s_x,480))
            hsvimg[rngy,rngx]  = scalepixel.(hsvimg[rngy,rngx],attn[LinearIndices((1:14,1:14))[i,j]])
        end
        display(hsvimg)
        display(RGB.(load("plots/$(k).png")))
    end
end

function scalepixel(pixel,scaler)
     return HSV(pixel.h,pixel.s,min(1.0,pixel.v+5*scaler))
end

function update_with_gclip(J, ws; clip=8.0f0)
    gnorm=zero(clip)
    for w in ws
        gnorm += Knet.norm(grad(J,w))^2
    end
    gnorm = sqrt(gnorm)
    if gnorm > clip
        for w in ws
            g = Knet.lmul!(clip/gnorm, grad(J,w))
            update!(w.value,g,w.opt)
        end
    else
        for w in ws
             update!(w.value,grad(J,w),w.opt)
        end
    end
end
