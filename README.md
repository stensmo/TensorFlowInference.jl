# TensorFlowInference.jl

Easy access to ready made TensorFlow models in Julia. Currently the focus is on running inference, loading a trained model is very easy, and use the inference results from Julia. Supports the latest TensorFlow versions (2.15), and can be made to support older versions, since the Julia code relies on the C API. 

Examples:
```
using TensorFlowInference
using MLDatasets 


testset = MNIST(:test)

testsetMatrix = testset.features


graph = TF_NewGraph();
status = TF_NewStatus();
saved_model_dir = "mnist/";


session = simpleLoadSessionFromSavedModel(saved_model_dir, graph, status)


printOutput(graph)

inputValues = testsetMatrix
graphInputName = "serving_default_flatten_1_input"
graphOutputName = "StatefulPartitionedCall"

outputMatrix = simpleSessionRun(session, graph, status, inputValues, graphInputName, graphOutputName)


```