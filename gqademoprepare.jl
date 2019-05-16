using Pkg; Pkg.activate(".")
include("src/main.jl")
include("configs/config3.jl")
using JSON

dhome    = o[:dhome]
trgthome = dhome * "gqa_demo/"
valfile  = dhome * "gen_gqa_objects_valInstances.json"
trnfile  = dhome * "gen_gqa_objects_trainInstances.json"
imgshome = dhome * "images/"
imgsdest = dhome * "gqa_demo/images/"
dicfile  = dhome * "wordembeddings_vocabs.jld2"

function getdemo(;set="train", total=100, featsize=(2048,100), fname="objects")
    println("feats are loading...")
    feats  = loadFeatures(dhome; featsize=featsize, fname=fname, h5=true)
    demofeats = Any[];
    println("data are loading...")
    data  = getQdata(dhome, set; fname=fname)
    demo  = data[randperm(length(data))[1:total]]
    qvoc,avoc,id2index = getDicts(dhome,nothing; fname=fname)
    println("Random selection are collecting...")
    for d in demo
        imgname = string(d["imageId"]["id"],".jpg")
        println("cp: ", imgshome*imgname, " dest:", imgsdest)
    	cp(imgshome*imgname, imgsdest*imgname;force=true)
        id = Int(id2index[d["imageId"]["id"]]["index"])
        push!(demofeats,feats[:,:,id])
    end
    demofeats = cat(demofeats...;dims=4)

    println("Demo feats are writed...")
    f = open(trgthome*"demo.bin","w")
    write(f,demofeats); close(f)

    println("Demo data is writed...")
    open(trgthome * "demo.json","w") do f
        write(f,json(demo))
    end
    KnetLayers.save(trgthome*"dict.jld2","word_dict",qvoc,"answer_dict",avoc,"id2index",id2index)
end

println("Creating demo folders if necessary...")
!isdir(trgthome) && mkdir(trgthome)
if isdir(imgsdest)
   rm(imgsdest;recursive=true)
end
@show !isdir(imgsdest) && mkdir(imgsdest)


getdemo(set="train")
run(`tar -cvzf gqa_demo.tar.gz $(trgthome)`)
