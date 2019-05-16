using Pkg; Pkg.activate("."); Pkg.instantiate(); #install required packages
include("gqademosetup.jl")
include("src/main.jl")

feats,qstsns,(qvoc,avoc) = loadDemoData("data/GQA2/gqa_demo/");
_,weights,o = loadmodel("models/gqaweights.jld2";onlywrun=true);
Mrun = copyto!(MACNetwork(o),weights)


rnd        = rand(1:length(qstsns)) # try 41
inst       = qstsns[rnd]
feat       = arrtype(permutedims(feats[:,:,rnd:rnd],(1,3,2)))
question   = map(x->get(qvoc,x,"<UNK>"),inst["question"]) 
answer     = get(avoc,inst["answer"],"<UNK>")
family     = 1
objectsNum = inst["objectsNum"]
results, prediction, interoutputs = singlerun(Mrun,feat,question;p=o[:p], objectnum=objectsNum);
answer==prediction[1]


img = load("data/GQA2/gqa_demo/images/$(inst["imageId"]["id"])")


textq  = qvoc[question];
println("Question: ",join(textq," "))
texta  = avoc[answer];
println("Answer: $(texta)\nPrediction: $(i2a[prediction]) ")


userinput = readline(stdin)
words = split(userinput) # tokenize(userinput)
question = [get(qvoc,wr,2) for wr in words]
results, prediction, interoutputs = singlerun(Mrun,feat,question;p=o[:p], objectnum=objectsNum);
println("Question: $(join(qvoc[question]," "))")
println("Prediction: $(avoc[prediction])")
