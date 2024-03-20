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

