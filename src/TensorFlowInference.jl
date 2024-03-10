module TensorFlowInference



if Sys.iswindows()
    const LibTensorFlow =  joinpath(dirname(pathof(TensorFlowInference)), "../lib/windows/tensorflow.dll")
end

if Sys.islinux()
    const LibTensorFlow =  joinpath(dirname(pathof(TensorFlowInference)), "../lib/linux/libtensorflow.so")
end

if Sys.isapple()
    const LibTensorFlow =  joinpath(dirname(pathof(TensorFlowInference)), "../lib/mac/libtensorflow.dylib")
end


include("TensorFlow.jl")
include("ConvenienceFunctions.jl")

export simpleLoadSessionFromSavedModel, simpleSessionRun, convertToTensor, convertFromTensor, printOutput

end # module TensorFlowInference