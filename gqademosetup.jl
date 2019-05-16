server="ai.ku.edu.tr/"
if !isdir("data/GQA2/gqa_demo")
    println("Downloading sample questions and images from CLEVR dataset...")
    download(server*"data/mac-network/gqa_demo.tar.gz","gqa_demo.tar.gz")
    run(`tar -xzf gqa_demo.tar.gz`)
    rm("gqa_demo.tar.gz")
end

if !isfile("models/gqaweights.jld2")
    println("Downloading pre-trained model from our servers...")
    download(server*"models/mac-network/gqaweights.jld2","models/gqaweights.jld2")
end
println("Demo setup is completed")
