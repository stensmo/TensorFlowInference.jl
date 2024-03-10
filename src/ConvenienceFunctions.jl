

function convertToTensor(inputData::AbstractArray)

    numDims = ndims(inputData)

    #=
    if numDims > 1
        inputDims = collect(1:numDims)
       
        inputDims[1] = 1
        inputDims[2] = 2
        
        inputDims[3] = 3
       
        
        #inputDims = reverse(inputDims)
        data = permutedims(inputData, Tuple(inputDims))
    else
        data = inputData
    end
    =#
    data = inputData

  
    len = convert(UInt64, sizeof(data))
    #println("len")
    #show(len)
    ndimsInt32::Int32 = convert(Int32,numDims);
    #println("ndimsInt32")
    #show(ndimsInt32)
    dims = reverse(collect(size(inputData)))
    #println("Dims")
    #show(dims)

    show(toTensorflowType[eltype(data)])
    input_tensor = TF_NewTensor(toTensorflowType[eltype(data)], dims, ndimsInt32, data,len , NoOpDeallocator, C_NULL);
    
    input_tensor == C_NULL && throw(ErrorException("Could not create tensor"))
        
    return input_tensor

end


function convertFromTensor(tensor::Ptr{TF_Tensor}) 
    tensor == C_NULL && return nothing
    dataTypeEnum = TF_TensorType(tensor)

    juliaType=tensorFlowTypes[Integer(dataTypeEnum)]

    convertedPointer = Base.unsafe_convert(Ptr{juliaType}, TF_TensorData(tensor))

    numDims = convert(Int64, TF_NumDims(tensor))
    

    dims = zeros(Int64, numDims)
    for i in 0:numDims-1
        dim = TF_Dim(tensor, Int32(i))
        dims[i+1] = convert(Int64, dim)
    end

    dimsTuple = Tuple(reverse(dims))
    println("Dims tuple")
    show(dimsTuple)

    finalBuffer = unsafe_wrap(Array{juliaType, numDims}, convertedPointer, dimsTuple)
    # Julia uses Column major order
 
    outputDims = collect(1:numDims)
    if numDims > 1
    
        outputDims = reverse(outputDims)
        permutedDims = permutedims(finalBuffer, Tuple(outputDims))

        return permutedDims
    end
    
    return finalBuffer

end



function checkStatusAndThrow(status)
    TF_GetCode(status) != TF_OK &&  throw(ErrorException(TF_Message(status)))
       
end


function simpleLoadSessionFromSavedModel(saved_model_dir, graph, status)
    sessionOpts = TF_NewSessionOptions();
   
    tags = ["serve"];
    ntags = convert(Int32, 1)
    
    metaGraphDef = TF_NewBuffer()

    session = TF_LoadSessionFromSavedModel(sessionOpts, C_NULL, saved_model_dir, tags, ntags, graph, metaGraphDef, status);

    metaBuf = TF_GetBuffer(metaGraphDef)

    checkStatusAndThrow(status)

    return session

end

function printOutput(graph)

    pos = Ref{Csize_t}(0)
    while (oper=TF_GraphNextOperation(graph, pos)) != C_NULL
    
        if TF_OperationOpType(oper) == "Placeholder" && TF_OperationName(oper) != "saver_filename"
                println("Output Name: $(TF_OperationName(oper))")
        
        end
    end
end

const tensorFlowTypes = [Float32, Float64, Int32, Int8, Int16, Int8, Cstring, ComplexF64, Int64, Bool, Int8, UInt8, Int32, Float16, Int16, Int16, UInt16]
    
const toTensorflowType = Dict{Any, TF_DataType}([(Float32, TF_FLOAT), (Float64,TF_DOUBLE), (Int32,TF_INT32),(UInt8,TF_UINT8), (Int16,TF_INT16 ), (Int8,TF_INT8 ), (string,TF_STRING ), (ComplexF64,TF_COMPLEX64 ), (Int64,TF_INT64  ), (Bool,TF_BOOL  )])


function noOpDeallocator(data, len, arg)

end

NoOpDeallocator = @cfunction(noOpDeallocator, Cvoid, (Ptr{Cvoid}, Csize_t, Ptr{Cvoid}))

function simpleSessionRun(session, graph, status, inputValues, graphInputName, graphOutputName)

    NumInputs = convert(Int32,1);

    t0 = TF_Output(TF_GraphOperationByName(graph, graphInputName), 0)
    
    t0.oper == C_NULL &&  throw(ErrorException("Cant find input"))

    Input = [t0];

    NumOutputs = convert(Int32,1);
 
    t2 = TF_Output(TF_GraphOperationByName(graph, graphOutputName), 0)

    t2.oper == C_NULL &&  throw(ErrorException("Cant find output"))
 
    Output = [t2];

    input_tensor=convertToTensor(inputValues)

    InputValues = [input_tensor];


    OutputValues = [Ptr{TF_Tensor}(0)];
    
    ntargets = convert(Cint, 0)
    # Run the Session
    TF_SessionRun(session, C_NULL, Input, InputValues, NumInputs, Output, OutputValues, NumOutputs, C_NULL, ntargets, C_NULL , status);

    checkStatusAndThrow(status)
    
    tensor = OutputValues[1]
    return convertFromTensor(tensor)
end










    
 