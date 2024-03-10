



to_c_type(t::Type) = t
to_c_type_pairs(va_list) = map(enumerate(to_c_type.(va_list))) do (ind, type)
    :(va_list[$ind]::$type)
end

# typedef void ( * TSL_PayloadVisitor ) ( const char * key , const char * value , void * capture )
const TSL_PayloadVisitor = Ptr{Cvoid}

struct TF_AllocatorAttributes
    struct_size::Csize_t
    on_host::Cuchar
end

@enum TF_AttrType::UInt32 begin
    TF_ATTR_STRING = 0
    TF_ATTR_INT = 1
    TF_ATTR_FLOAT = 2
    TF_ATTR_BOOL = 3
    TF_ATTR_TYPE = 4
    TF_ATTR_SHAPE = 5
    TF_ATTR_TENSOR = 6
    TF_ATTR_PLACEHOLDER = 7
    TF_ATTR_FUNC = 8
end

struct TF_Buffer
    data::Ptr{Cvoid}
    length::Csize_t
    data_deallocator::Ptr{Cvoid}
end

function TF_NewBufferFromString(proto, proto_len::Csize_t)
    ccall((:TF_NewBufferFromString, LibTensorFlow), Ptr{TF_Buffer}, (Ptr{Cvoid}, Csize_t), proto, proto_len)
end

function TF_NewBuffer()
    ccall((:TF_NewBuffer, LibTensorFlow), Ptr{TF_Buffer}, ())
end

function TF_DeleteBuffer(arg1)
    ccall((:TF_DeleteBuffer, LibTensorFlow), Cvoid, (Ptr{TF_Buffer},), arg1)
end

function TF_GetBuffer(buffer)
    ccall((:TF_GetBuffer, LibTensorFlow), TF_Buffer, (Ptr{TF_Buffer},), buffer)
end


@enum TF_DataType::UInt32 begin
    TF_FLOAT = 1
    TF_DOUBLE = 2
    TF_INT32 = 3
    TF_UINT8 = 4
    TF_INT16 = 5
    TF_INT8 = 6
    TF_STRING = 7
    TF_COMPLEX64 = 8
    # TF_COMPLEX = 8
    TF_INT64 = 9
    TF_BOOL = 10
    TF_QINT8 = 11
    TF_QUINT8 = 12
    TF_QINT32 = 13
    TF_BFLOAT16 = 14
    TF_QINT16 = 15
    TF_QUINT16 = 16
    TF_UINT16 = 17
    TF_COMPLEX128 = 18
    TF_HALF = 19
    TF_RESOURCE = 20
    TF_VARIANT = 21
    TF_UINT32 = 22
    TF_UINT64 = 23
    TF_FLOAT8_E5M2 = 24
    TF_FLOAT8_E4M3FN = 25
    TF_INT4 = 29
    TF_UINT4 = 30
end

function TF_DataTypeSize(dt::TF_DataType)
    ccall((:TF_DataTypeSize, LibTensorFlow), Csize_t, (TF_DataType,), dt)
end

mutable struct TF_Tensor end

mutable struct TSL_Status end

@enum TSL_Code::UInt32 begin
    TSL_OK = 0
    TSL_CANCELLED = 1
    TSL_UNKNOWN = 2
    TSL_INVALID_ARGUMENT = 3
    TSL_DEADLINE_EXCEEDED = 4
    TSL_NOT_FOUND = 5
    TSL_ALREADY_EXISTS = 6
    TSL_PERMISSION_DENIED = 7
    TSL_UNAUTHENTICATED = 16
    TSL_RESOURCE_EXHAUSTED = 8
    TSL_FAILED_PRECONDITION = 9
    TSL_ABORTED = 10
    TSL_OUT_OF_RANGE = 11
    TSL_UNIMPLEMENTED = 12
    TSL_INTERNAL = 13
    TSL_UNAVAILABLE = 14
    TSL_DATA_LOSS = 15
end

function TSL_NewStatus()
    ccall((:TSL_NewStatus, LibTensorFlow), Ptr{TSL_Status}, ())
end

function TSL_DeleteStatus(arg1)
    ccall((:TSL_DeleteStatus, LibTensorFlow), Cvoid, (Ptr{TSL_Status},), arg1)
end

function TSL_SetStatus(s, code::TSL_Code, msg)
    ccall((:TSL_SetStatus, LibTensorFlow), Cvoid, (Ptr{TSL_Status}, TSL_Code, Ptr{Cchar}), s, code, msg)
end

function TSL_SetPayload(s, key, value)
    ccall((:TSL_SetPayload, LibTensorFlow), Cvoid, (Ptr{TSL_Status}, Ptr{Cchar}, Ptr{Cchar}), s, key, value)
end

function TSL_ForEachPayload(s, visitor::TSL_PayloadVisitor, capture)
    ccall((:TSL_ForEachPayload, LibTensorFlow), Cvoid, (Ptr{TSL_Status}, TSL_PayloadVisitor, Ptr{Cvoid}), s, visitor, capture)
end

function TSL_SetStatusFromIOError(s, error_code::Cint, context)
    ccall((:TSL_SetStatusFromIOError, LibTensorFlow), Cvoid, (Ptr{TSL_Status}, Cint, Ptr{Cchar}), s, error_code, context)
end

function TSL_GetCode(s)
    ccall((:TSL_GetCode, LibTensorFlow), TSL_Code, (Ptr{TSL_Status},), s)
end

function TSL_Message(s)
    ccall((:TSL_Message, LibTensorFlow), Ptr{Cchar}, (Ptr{TSL_Status},), s)
end

const TF_Status = TSL_Status

const TF_Code = TSL_Code

function TF_NewStatus()
    ccall((:TF_NewStatus, LibTensorFlow), Ptr{TF_Status}, ())
end

function TF_DeleteStatus(arg1)
    ccall((:TF_DeleteStatus, LibTensorFlow), Cvoid, (Ptr{TF_Status},), arg1)
end

function TF_SetStatus(s, code::TF_Code, msg)
    ccall((:TF_SetStatus, LibTensorFlow), Cvoid, (Ptr{TF_Status}, TF_Code, Ptr{Cchar}), s, code, msg)
end

function TF_SetPayload(s, key, value)
    ccall((:TF_SetPayload, LibTensorFlow), Cvoid, (Ptr{TF_Status}, Ptr{Cchar}, Ptr{Cchar}), s, key, value)
end

function TF_ForEachPayload(s, visitor::TSL_PayloadVisitor, capture)
    ccall((:TF_ForEachPayload, LibTensorFlow), Cvoid, (Ptr{TF_Status}, TSL_PayloadVisitor, Ptr{Cvoid}), s, visitor, capture)
end

function TF_SetStatusFromIOError(s, error_code::Cint, context)
    ccall((:TF_SetStatusFromIOError, LibTensorFlow), Cvoid, (Ptr{TF_Status}, Cint, Ptr{Cchar}), s, error_code, context)
end

function TF_GetCode(s)
    ccall((:TF_GetCode, LibTensorFlow), TF_Code, (Ptr{TF_Status},), s)
end

function TF_Message(s)::AbstractString
    unsafe_string(ccall((:TF_Message, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_Status},), s))
end



function TF_NewTensor(arg1::TF_DataType, dims, num_dims::Cint, data, len::Csize_t, deallocator, deallocator_arg)
    ccall((:TF_NewTensor, LibTensorFlow), Ptr{TF_Tensor}, (TF_DataType, Ptr{Int64}, Cint, Ptr{Cvoid}, Csize_t, Ptr{Cvoid}, Ptr{Cvoid}), arg1, dims, num_dims, data, len, deallocator, deallocator_arg)
end

function TF_AllocateTensor(arg1::TF_DataType, dims, num_dims::Cint, len::Csize_t)
    ccall((:TF_AllocateTensor, LibTensorFlow), Ptr{TF_Tensor}, (TF_DataType, Ptr{Int64}, Cint, Csize_t), arg1, dims, num_dims, len)
end

function TF_TensorMaybeMove(tensor)
    ccall((:TF_TensorMaybeMove, LibTensorFlow), Ptr{TF_Tensor}, (Ptr{TF_Tensor},), tensor)
end

function TF_DeleteTensor(arg1)
    ccall((:TF_DeleteTensor, LibTensorFlow), Cvoid, (Ptr{TF_Tensor},), arg1)
end

function TF_TensorType(arg1)
    ccall((:TF_TensorType, LibTensorFlow), TF_DataType, (Ptr{TF_Tensor},), arg1)
end

function TF_SetShape(tensor, dims, num_dims::Cint)
    ccall((:TF_SetShape, LibTensorFlow), Cvoid, (Ptr{TF_Tensor}, Ptr{Int64}, Cint), tensor, dims, num_dims)
end

function TF_NumDims(arg1)
    ccall((:TF_NumDims, LibTensorFlow), Cint, (Ptr{TF_Tensor},), arg1)
end

function TF_Dim(tensor, dim_index::Cint)
    ccall((:TF_Dim, LibTensorFlow), Int64, (Ptr{TF_Tensor}, Cint), tensor, dim_index)
end

function TF_TensorByteSize(arg1)
    ccall((:TF_TensorByteSize, LibTensorFlow), Csize_t, (Ptr{TF_Tensor},), arg1)
end

function TF_TensorData(arg1)
    ccall((:TF_TensorData, LibTensorFlow), Ptr{Cvoid}, (Ptr{TF_Tensor},), arg1)
end

function TF_TensorElementCount(tensor)
    ccall((:TF_TensorElementCount, LibTensorFlow), Int64, (Ptr{TF_Tensor},), tensor)
end

function TF_TensorBitcastFrom(from, type::TF_DataType, to, new_dims, num_new_dims::Cint, status)
    ccall((:TF_TensorBitcastFrom, LibTensorFlow), Cvoid, (Ptr{TF_Tensor}, TF_DataType, Ptr{TF_Tensor}, Ptr{Int64}, Cint, Ptr{TF_Status}), from, type, to, new_dims, num_new_dims, status)
end

function TF_TensorIsAligned(arg1)
    ccall((:TF_TensorIsAligned, LibTensorFlow), Bool, (Ptr{TF_Tensor},), arg1)
end

function TF_swap32(host_int::UInt32)
    ccall((:TF_swap32, LibTensorFlow), UInt32, (UInt32,), host_int)
end

function TF_align16(i::Csize_t)
    ccall((:TF_align16, LibTensorFlow), Csize_t, (Csize_t,), i)
end

function TF_max(a::Csize_t, b::Csize_t)
    ccall((:TF_max, LibTensorFlow), Csize_t, (Csize_t, Csize_t), a, b)
end

function TF_min(a::Csize_t, b::Csize_t)
    ccall((:TF_min, LibTensorFlow), Csize_t, (Csize_t, Csize_t), a, b)
end

@enum TF_TString_Type::UInt32 begin
    TF_TSTR_SMALL = 0
    TF_TSTR_LARGE = 1
    TF_TSTR_OFFSET = 2
    TF_TSTR_VIEW = 3
    # TF_TSTR_TYPE_MASK = 3
end

struct TF_TString_Large
    size::Csize_t
    cap::Csize_t
    ptr::Ptr{Cchar}
end

struct TF_TString_Offset
    size::UInt32
    offset::UInt32
    count::UInt32
end

struct TF_TString_View
    size::Csize_t
    ptr::Ptr{Cchar}
end

struct TF_TString_Raw
    raw::NTuple{24, UInt8}
end

struct TF_TString_Union
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{TF_TString_Union}, f::Symbol)
    f === :large && return Ptr{TF_TString_Large}(x + 0)
    f === :offset && return Ptr{TF_TString_Offset}(x + 0)
    f === :view && return Ptr{TF_TString_View}(x + 0)
    f === :raw && return Ptr{TF_TString_Raw}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::TF_TString_Union, f::Symbol)
    r = Ref{TF_TString_Union}(x)
    ptr = Base.unsafe_convert(Ptr{TF_TString_Union}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{TF_TString_Union}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

@enum __JL_Ctag_8::UInt32 begin
    TF_TString_SmallCapacity = 22
end

struct TF_TString_Small
    size::UInt8
    str::NTuple{23, Cchar}
end

struct __JL_Ctag_9
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{__JL_Ctag_9}, f::Symbol)
    f === :smll && return Ptr{TF_TString_Small}(x + 0)
    f === :large && return Ptr{TF_TString_Large}(x + 0)
    f === :offset && return Ptr{TF_TString_Offset}(x + 0)
    f === :view && return Ptr{TF_TString_View}(x + 0)
    f === :raw && return Ptr{TF_TString_Raw}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::__JL_Ctag_9, f::Symbol)
    r = Ref{__JL_Ctag_9}(x)
    ptr = Base.unsafe_convert(Ptr{__JL_Ctag_9}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{__JL_Ctag_9}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

struct TF_TString
    data::NTuple{24, UInt8}
end

function Base.getproperty(x::Ptr{TF_TString}, f::Symbol)
    f === :u && return Ptr{__JL_Ctag_9}(x + 0)
    return getfield(x, f)
end

function Base.getproperty(x::TF_TString, f::Symbol)
    r = Ref{TF_TString}(x)
    ptr = Base.unsafe_convert(Ptr{TF_TString}, r)
    fptr = getproperty(ptr, f)
    GC.@preserve r unsafe_load(fptr)
end

function Base.setproperty!(x::Ptr{TF_TString}, f::Symbol, v)
    unsafe_store!(getproperty(x, f), v)
end

function TF_TString_GetType(str)
    ccall((:TF_TString_GetType, LibTensorFlow), TF_TString_Type, (Ptr{TF_TString},), str)
end

function TF_TString_ToActualSizeT(size::Csize_t)
    ccall((:TF_TString_ToActualSizeT, LibTensorFlow), Csize_t, (Csize_t,), size)
end

function TF_TString_ToInternalSizeT(size::Csize_t, type::TF_TString_Type)
    ccall((:TF_TString_ToInternalSizeT, LibTensorFlow), Csize_t, (Csize_t, TF_TString_Type), size, type)
end

function TF_TString_Init(str)
    ccall((:TF_TString_Init, LibTensorFlow), Cvoid, (Ptr{TF_TString},), str)
end

function TF_TString_Dealloc(str)
    ccall((:TF_TString_Dealloc, LibTensorFlow), Cvoid, (Ptr{TF_TString},), str)
end

function TF_TString_GetSize(str)
    ccall((:TF_TString_GetSize, LibTensorFlow), Csize_t, (Ptr{TF_TString},), str)
end

function TF_TString_GetCapacity(str)
    ccall((:TF_TString_GetCapacity, LibTensorFlow), Csize_t, (Ptr{TF_TString},), str)
end

function TF_TString_GetDataPointer(str)
    ccall((:TF_TString_GetDataPointer, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_TString},), str)
end

function TF_TString_ResizeUninitialized(str, new_size::Csize_t)
    ccall((:TF_TString_ResizeUninitialized, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_TString}, Csize_t), str, new_size)
end

function TF_TString_GetMutableDataPointer(str)
    ccall((:TF_TString_GetMutableDataPointer, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_TString},), str)
end

function TF_TString_Reserve(str, new_cap::Csize_t)
    ccall((:TF_TString_Reserve, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Csize_t), str, new_cap)
end

function TF_TString_ReserveAmortized(str, new_cap::Csize_t)
    ccall((:TF_TString_ReserveAmortized, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Csize_t), str, new_cap)
end

function TF_TString_Resize(str, new_size::Csize_t, c::Cchar)
    ccall((:TF_TString_Resize, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_TString}, Csize_t, Cchar), str, new_size, c)
end

function TF_TString_AssignView(dst, src, size::Csize_t)
    ccall((:TF_TString_AssignView, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{Cchar}, Csize_t), dst, src, size)
end

function TF_TString_AppendN(dst, src, src_size::Csize_t)
    ccall((:TF_TString_AppendN, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{Cchar}, Csize_t), dst, src, src_size)
end

function TF_TString_Append(dst, src)
    ccall((:TF_TString_Append, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{TF_TString}), dst, src)
end

function TF_TString_Copy(dst, src, size::Csize_t)
    ccall((:TF_TString_Copy, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{Cchar}, Csize_t), dst, src, size)
end

function TF_TString_Assign(dst, src)
    ccall((:TF_TString_Assign, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{TF_TString}), dst, src)
end

function TF_TString_Move(dst, src)
    ccall((:TF_TString_Move, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{TF_TString}), dst, src)
end

function TF_StringInit(t)
    ccall((:TF_StringInit, LibTensorFlow), Cvoid, (Ptr{TF_TString},), t)
end

function TF_StringCopy(dst, src, size::Csize_t)
    ccall((:TF_StringCopy, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{Cchar}, Csize_t), dst, src, size)
end

function TF_StringAssignView(dst, src, size::Csize_t)
    ccall((:TF_StringAssignView, LibTensorFlow), Cvoid, (Ptr{TF_TString}, Ptr{Cchar}, Csize_t), dst, src, size)
end

function TF_StringGetDataPointer(tstr)
    ccall((:TF_StringGetDataPointer, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_TString},), tstr)
end

function TF_StringGetType(str)
    ccall((:TF_StringGetType, LibTensorFlow), TF_TString_Type, (Ptr{TF_TString},), str)
end

function TF_StringGetSize(tstr)
    ccall((:TF_StringGetSize, LibTensorFlow), Csize_t, (Ptr{TF_TString},), tstr)
end

function TF_StringGetCapacity(str)
    ccall((:TF_StringGetCapacity, LibTensorFlow), Csize_t, (Ptr{TF_TString},), str)
end

function TF_StringDealloc(tstr)
    ccall((:TF_StringDealloc, LibTensorFlow), Cvoid, (Ptr{TF_TString},), tstr)
end

function TF_Version()
    ccall((:TF_Version, LibTensorFlow), Ptr{Cchar}, ())
end

function TF_TensorFromProto(from, to, status)
    ccall((:TF_TensorFromProto, LibTensorFlow), Cvoid, (Ptr{TF_Buffer}, Ptr{TF_Tensor}, Ptr{TF_Status}), from, to, status)
end

struct TF_StringView
    data::Ptr{Cchar}
    len::Csize_t
end

mutable struct TF_SessionOptions end

function TF_NewSessionOptions()
    ccall((:TF_NewSessionOptions, LibTensorFlow), Ptr{TF_SessionOptions}, ())
end

function TF_SetTarget(options, target)
    ccall((:TF_SetTarget, LibTensorFlow), Cvoid, (Ptr{TF_SessionOptions}, Ptr{Cchar}), options, target)
end

function TF_SetConfig(options, proto, proto_len::Csize_t, status)
    ccall((:TF_SetConfig, LibTensorFlow), Cvoid, (Ptr{TF_SessionOptions}, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), options, proto, proto_len, status)
end

function TF_DeleteSessionOptions(arg1)
    ccall((:TF_DeleteSessionOptions, LibTensorFlow), Cvoid, (Ptr{TF_SessionOptions},), arg1)
end

mutable struct TF_Graph end

function TF_NewGraph()
    ccall((:TF_NewGraph, LibTensorFlow), Ptr{TF_Graph}, ())
end

function TF_DeleteGraph(arg1)
    ccall((:TF_DeleteGraph, LibTensorFlow), Cvoid, (Ptr{TF_Graph},), arg1)
end

mutable struct TF_OperationDescription end

mutable struct TF_Operation end

struct TF_Input
    oper::Ptr{TF_Operation}
    index::Cint
end

struct TF_Output
    oper::Ptr{TF_Operation}
    index::Cint
end

mutable struct TF_Function end

mutable struct TF_FunctionOptions end

function TF_GraphSetTensorShape(graph::Ptr{TF_Graph}, output::TF_Output, dims, num_dims::Cint, status)
    ccall((:TF_GraphSetTensorShape, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, TF_Output, Ptr{Int64}, Cint, Ptr{TF_Status}), graph, output, dims, num_dims, status)
end

function TF_GraphGetTensorNumDims(graph::Ptr{TF_Graph}, output::TF_Output, status)
    ccall((:TF_GraphGetTensorNumDims, LibTensorFlow), Cint, (Ptr{TF_Graph}, TF_Output, Ptr{TF_Status}), graph, output, status)
end
function TF_GraphGetTensorShape(graph::Ptr{TF_Graph}, output::TF_Output, dims, num_dims::Cint, status)
    ccall((:TF_GraphGetTensorShape, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, TF_Output, Ptr{Int64}, Cint, Ptr{TF_Status}), graph, output, dims, num_dims, status)
end

function TF_NewOperationLocked(graph::Ptr{TF_Graph}, op_type, oper_name)
    ccall((:TF_NewOperationLocked, LibTensorFlow), Ptr{TF_OperationDescription}, (Ptr{TF_Graph}, Ptr{Cchar}, Ptr{Cchar}), graph, op_type, oper_name)
end

function TF_NewOperation(graph::Ptr{TF_Graph}, op_type, oper_name)
    ccall((:TF_NewOperation, LibTensorFlow), Ptr{TF_OperationDescription}, (Ptr{TF_Graph}, Ptr{Cchar}, Ptr{Cchar}), graph, op_type, oper_name)
end

function TF_SetDevice(desc, device)
    ccall((:TF_SetDevice, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}), desc, device)
end

function TF_AddInput(desc, input::TF_Output)
    ccall((:TF_AddInput, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, TF_Output), desc, input)
end

function TF_AddInputList(desc, inputs, num_inputs::Cint)
    ccall((:TF_AddInputList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{TF_Output}, Cint), desc, inputs, num_inputs)
end

function TF_AddControlInput(desc, input)
    ccall((:TF_AddControlInput, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{TF_Operation}), desc, input)
end

function TF_ColocateWith(desc, op)
    ccall((:TF_ColocateWith, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{TF_Operation}), desc, op)
end

function TF_SetAttrString(desc, attr_name, value, length::Csize_t)
    ccall((:TF_SetAttrString, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Cvoid}, Csize_t), desc, attr_name, value, length)
end

function TF_SetAttrStringList(desc, attr_name, values, lengths, num_values::Cint)
    ccall((:TF_SetAttrStringList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Cint), desc, attr_name, values, lengths, num_values)
end

function TF_SetAttrInt(desc, attr_name, value::Int64)
    ccall((:TF_SetAttrInt, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Int64), desc, attr_name, value)
end

function TF_SetAttrIntList(desc, attr_name, values, num_values::Cint)
    ccall((:TF_SetAttrIntList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Int64}, Cint), desc, attr_name, values, num_values)
end

function TF_SetAttrFloat(desc, attr_name, value::Cfloat)
    ccall((:TF_SetAttrFloat, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Cfloat), desc, attr_name, value)
end

function TF_SetAttrFloatList(desc, attr_name, values, num_values::Cint)
    ccall((:TF_SetAttrFloatList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Cfloat}, Cint), desc, attr_name, values, num_values)
end

function TF_SetAttrBool(desc, attr_name, value::Cuchar)
    ccall((:TF_SetAttrBool, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Cuchar), desc, attr_name, value)
end

function TF_SetAttrBoolList(desc, attr_name, values, num_values::Cint)
    ccall((:TF_SetAttrBoolList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Cuchar}, Cint), desc, attr_name, values, num_values)
end

function TF_SetAttrType(desc, attr_name, value::TF_DataType)
    ccall((:TF_SetAttrType, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, TF_DataType), desc, attr_name, value)
end

function TF_SetAttrTypeList(desc, attr_name, values, num_values::Cint)
    ccall((:TF_SetAttrTypeList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{TF_DataType}, Cint), desc, attr_name, values, num_values)
end

function TF_SetAttrPlaceholder(desc, attr_name, placeholder)
    ccall((:TF_SetAttrPlaceholder, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Cchar}), desc, attr_name, placeholder)
end

function TF_SetAttrFuncName(desc, attr_name, value, length::Csize_t)
    ccall((:TF_SetAttrFuncName, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Cchar}, Csize_t), desc, attr_name, value, length)
end

function TF_SetAttrShape(desc, attr_name, dims, num_dims::Cint)
    ccall((:TF_SetAttrShape, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Int64}, Cint), desc, attr_name, dims, num_dims)
end

function TF_SetAttrShapeList(desc, attr_name, dims, num_dims, num_shapes::Cint)
    ccall((:TF_SetAttrShapeList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Ptr{Int64}}, Ptr{Cint}, Cint), desc, attr_name, dims, num_dims, num_shapes)
end

function TF_SetAttrTensorShapeProto(desc, attr_name, proto, proto_len::Csize_t, status)
    ccall((:TF_SetAttrTensorShapeProto, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), desc, attr_name, proto, proto_len, status)
end

function TF_SetAttrTensorShapeProtoList(desc, attr_name, protos, proto_lens, num_shapes::Cint, status)
    ccall((:TF_SetAttrTensorShapeProtoList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Cint, Ptr{TF_Status}), desc, attr_name, protos, proto_lens, num_shapes, status)
end

function TF_SetAttrTensor(desc, attr_name, value, status)
    ccall((:TF_SetAttrTensor, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{TF_Tensor}, Ptr{TF_Status}), desc, attr_name, value, status)
end

function TF_SetAttrTensorList(desc, attr_name, values, num_values::Cint, status)
    ccall((:TF_SetAttrTensorList, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{TF_Status}), desc, attr_name, values, num_values, status)
end

function TF_SetAttrValueProto(desc, attr_name, proto, proto_len::Csize_t, status)
    ccall((:TF_SetAttrValueProto, LibTensorFlow), Cvoid, (Ptr{TF_OperationDescription}, Ptr{Cchar}, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), desc, attr_name, proto, proto_len, status)
end

function TF_FinishOperationLocked(desc, status)
    ccall((:TF_FinishOperationLocked, LibTensorFlow), Ptr{TF_Operation}, (Ptr{TF_OperationDescription}, Ptr{TF_Status}), desc, status)
end

function TF_FinishOperation(desc, status)
    ccall((:TF_FinishOperation, LibTensorFlow), Ptr{TF_Operation}, (Ptr{TF_OperationDescription}, Ptr{TF_Status}), desc, status)
end

function TF_OperationName(oper)
    unsafe_string(ccall((:TF_OperationName, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_Operation},), oper))
end

function TF_OperationOpType(oper)
    unsafe_string(ccall((:TF_OperationOpType, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_Operation},), oper))
end

function TF_OperationDevice(oper)
    unsafe_string(ccall((:TF_OperationDevice, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_Operation},), oper))
end

function TF_OperationNumOutputs(oper)
    ccall((:TF_OperationNumOutputs, LibTensorFlow), Cint, (Ptr{TF_Operation},), oper)
end

function TF_OperationOutputType(oper_out::TF_Output)
    ccall((:TF_OperationOutputType, LibTensorFlow), TF_DataType, (TF_Output,), oper_out)
end

function TF_OperationOutputListLength(oper, arg_name, status)
    ccall((:TF_OperationOutputListLength, LibTensorFlow), Cint, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_Status}), oper, arg_name, status)
end

function TF_OperationNumInputs(oper)
    ccall((:TF_OperationNumInputs, LibTensorFlow), Cint, (Ptr{TF_Operation},), oper)
end

function TF_OperationInputType(oper_in::TF_Input)
    ccall((:TF_OperationInputType, LibTensorFlow), TF_DataType, (TF_Input,), oper_in)
end

function TF_OperationInputListLength(oper, arg_name, status)
    ccall((:TF_OperationInputListLength, LibTensorFlow), Cint, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_Status}), oper, arg_name, status)
end

function TF_OperationInput(oper_in::TF_Input)
    ccall((:TF_OperationInput, LibTensorFlow), TF_Output, (TF_Input,), oper_in)
end

function TF_OperationAllInputs(oper, inputs, max_inputs::Cint)
    ccall((:TF_OperationAllInputs, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{TF_Output}, Cint), oper, inputs, max_inputs)
end

function TF_OperationOutputNumConsumers(oper_out::TF_Output)
    ccall((:TF_OperationOutputNumConsumers, LibTensorFlow), Cint, (TF_Output,), oper_out)
end

function TF_OperationOutputConsumers(oper_out::TF_Output, consumers, max_consumers::Cint)
    ccall((:TF_OperationOutputConsumers, LibTensorFlow), Cint, (TF_Output, Ptr{TF_Input}, Cint), oper_out, consumers, max_consumers)
end

function TF_OperationNumControlInputs(oper)
    ccall((:TF_OperationNumControlInputs, LibTensorFlow), Cint, (Ptr{TF_Operation},), oper)
end

function TF_OperationGetControlInputs(oper, control_inputs, max_control_inputs::Cint)
    ccall((:TF_OperationGetControlInputs, LibTensorFlow), Cint, (Ptr{TF_Operation}, Ptr{Ptr{TF_Operation}}, Cint), oper, control_inputs, max_control_inputs)
end

function TF_OperationNumControlOutputs(oper)
    ccall((:TF_OperationNumControlOutputs, LibTensorFlow), Cint, (Ptr{TF_Operation},), oper)
end

function TF_OperationGetControlOutputs(oper, control_outputs, max_control_outputs::Cint)
    ccall((:TF_OperationGetControlOutputs, LibTensorFlow), Cint, (Ptr{TF_Operation}, Ptr{Ptr{TF_Operation}}, Cint), oper, control_outputs, max_control_outputs)
end

struct TF_AttrMetadata
    is_list::Cuchar
    list_size::Int64
    type::TF_AttrType
    total_size::Int64
end

function TF_OperationGetAttrMetadata(oper, attr_name, status)
    ccall((:TF_OperationGetAttrMetadata, LibTensorFlow), TF_AttrMetadata, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_Status}), oper, attr_name, status)
end

function TF_OperationGetAttrString(oper, attr_name, value, max_length::Csize_t, status)
    ccall((:TF_OperationGetAttrString, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), oper, attr_name, value, max_length, status)
end

function TF_OperationGetAttrStringList(oper, attr_name, values, lengths, max_values::Cint, storage, storage_size::Csize_t, status)
    ccall((:TF_OperationGetAttrStringList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Ptr{Cvoid}}, Ptr{Csize_t}, Cint, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), oper, attr_name, values, lengths, max_values, storage, storage_size, status)
end

function TF_OperationGetAttrInt(oper, attr_name, value, status)
    ccall((:TF_OperationGetAttrInt, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Int64}, Ptr{TF_Status}), oper, attr_name, value, status)
end

function TF_OperationGetAttrIntList(oper, attr_name, values, max_values::Cint, status)
    ccall((:TF_OperationGetAttrIntList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Int64}, Cint, Ptr{TF_Status}), oper, attr_name, values, max_values, status)
end

function TF_OperationGetAttrFloat(oper, attr_name, value, status)
    ccall((:TF_OperationGetAttrFloat, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Cfloat}, Ptr{TF_Status}), oper, attr_name, value, status)
end

function TF_OperationGetAttrFloatList(oper, attr_name, values, max_values::Cint, status)
    ccall((:TF_OperationGetAttrFloatList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Cfloat}, Cint, Ptr{TF_Status}), oper, attr_name, values, max_values, status)
end

function TF_OperationGetAttrBool(oper, attr_name, value, status)
    ccall((:TF_OperationGetAttrBool, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Cuchar}, Ptr{TF_Status}), oper, attr_name, value, status)
end

function TF_OperationGetAttrBoolList(oper, attr_name, values, max_values::Cint, status)
    ccall((:TF_OperationGetAttrBoolList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Cuchar}, Cint, Ptr{TF_Status}), oper, attr_name, values, max_values, status)
end

function TF_OperationGetAttrType(oper, attr_name, value, status)
    ccall((:TF_OperationGetAttrType, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_DataType}, Ptr{TF_Status}), oper, attr_name, value, status)
end

function TF_OperationGetAttrTypeList(oper, attr_name, values, max_values::Cint, status)
    ccall((:TF_OperationGetAttrTypeList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_DataType}, Cint, Ptr{TF_Status}), oper, attr_name, values, max_values, status)
end

function TF_OperationGetAttrShape(oper, attr_name, value, num_dims::Cint, status)
    ccall((:TF_OperationGetAttrShape, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Int64}, Cint, Ptr{TF_Status}), oper, attr_name, value, num_dims, status)
end

function TF_OperationGetAttrShapeList(oper, attr_name, dims, num_dims, num_shapes::Cint, storage, storage_size::Cint, status)
    ccall((:TF_OperationGetAttrShapeList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Ptr{Int64}}, Ptr{Cint}, Cint, Ptr{Int64}, Cint, Ptr{TF_Status}), oper, attr_name, dims, num_dims, num_shapes, storage, storage_size, status)
end

function TF_OperationGetAttrTensorShapeProto(oper, attr_name, value, status)
    ccall((:TF_OperationGetAttrTensorShapeProto, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_Buffer}, Ptr{TF_Status}), oper, attr_name, value, status)
end

function TF_OperationGetAttrTensorShapeProtoList(oper, attr_name, values, max_values::Cint, status)
    ccall((:TF_OperationGetAttrTensorShapeProtoList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Ptr{TF_Buffer}}, Cint, Ptr{TF_Status}), oper, attr_name, values, max_values, status)
end

function TF_OperationGetAttrTensor(oper, attr_name, value, status)
    ccall((:TF_OperationGetAttrTensor, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Ptr{TF_Tensor}}, Ptr{TF_Status}), oper, attr_name, value, status)
end

function TF_OperationGetAttrTensorList(oper, attr_name, values, max_values::Cint, status)
    ccall((:TF_OperationGetAttrTensorList, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{TF_Status}), oper, attr_name, values, max_values, status)
end

function TF_OperationGetAttrValueProto(oper, attr_name, output_attr_value, status)
    ccall((:TF_OperationGetAttrValueProto, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_Buffer}, Ptr{TF_Status}), oper, attr_name, output_attr_value, status)
end

function TF_OperationGetNumAttrs(oper)
    ccall((:TF_OperationGetNumAttrs, LibTensorFlow), Cint, (Ptr{TF_Operation},), oper)
end

function TF_OperationGetAttrNameLength(oper, i::Cint)
    ccall((:TF_OperationGetAttrNameLength, LibTensorFlow), Cint, (Ptr{TF_Operation}, Cint), oper, i)
end

function TF_OperationGetAttrName(oper, i::Cint, output, status)
    ccall((:TF_OperationGetAttrName, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Cint, Ptr{Cchar}, Ptr{TF_Status}), oper, i, output, status)
end

function TF_GraphOperationByName(graph::Ptr{TF_Graph}, oper_name)
    ccall((:TF_GraphOperationByName, LibTensorFlow), Ptr{TF_Operation}, (Ptr{TF_Graph}, Ptr{Cchar}), graph, oper_name)
end

function TF_GraphNextOperation(graph::Ptr{TF_Graph}, pos)
    ccall((:TF_GraphNextOperation, LibTensorFlow), Ptr{TF_Operation}, (Ptr{TF_Graph}, Ptr{Csize_t}), graph, pos)
end

function TF_GraphToGraphDef(graph::Ptr{TF_Graph}, output_graph_def, status)
    ccall((:TF_GraphToGraphDef, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Buffer}, Ptr{TF_Status}), graph, output_graph_def, status)
end

function TF_GraphGetOpDef(graph::Ptr{TF_Graph}, op_name, output_op_def, status)
    ccall((:TF_GraphGetOpDef, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{Cchar}, Ptr{TF_Buffer}, Ptr{TF_Status}), graph, op_name, output_op_def, status)
end

function TF_GraphVersions(graph::Ptr{TF_Graph}, output_version_def, status)
    ccall((:TF_GraphVersions, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Buffer}, Ptr{TF_Status}), graph, output_version_def, status)
end

mutable struct TF_ImportGraphDefOptions end

function TF_NewImportGraphDefOptions()
    ccall((:TF_NewImportGraphDefOptions, LibTensorFlow), Ptr{TF_ImportGraphDefOptions}, ())
end

function TF_DeleteImportGraphDefOptions(opts)
    ccall((:TF_DeleteImportGraphDefOptions, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions},), opts)
end

function TF_ImportGraphDefOptionsSetPrefix(opts, prefix)
    ccall((:TF_ImportGraphDefOptionsSetPrefix, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Ptr{Cchar}), opts, prefix)
end

function TF_ImportGraphDefOptionsSetDefaultDevice(opts, device)
    ccall((:TF_ImportGraphDefOptionsSetDefaultDevice, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Ptr{Cchar}), opts, device)
end

function TF_ImportGraphDefOptionsSetUniquifyNames(opts, uniquify_names::Cuchar)
    ccall((:TF_ImportGraphDefOptionsSetUniquifyNames, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Cuchar), opts, uniquify_names)
end

function TF_ImportGraphDefOptionsSetUniquifyPrefix(opts, uniquify_prefix::Cuchar)
    ccall((:TF_ImportGraphDefOptionsSetUniquifyPrefix, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Cuchar), opts, uniquify_prefix)
end

function TF_ImportGraphDefOptionsAddInputMapping(opts, src_name, src_index::Cint, dst::TF_Output)
    ccall((:TF_ImportGraphDefOptionsAddInputMapping, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Ptr{Cchar}, Cint, TF_Output), opts, src_name, src_index, dst)
end

function TF_ImportGraphDefOptionsRemapControlDependency(opts, src_name, dst)
    ccall((:TF_ImportGraphDefOptionsRemapControlDependency, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Ptr{Cchar}, Ptr{TF_Operation}), opts, src_name, dst)
end

function TF_ImportGraphDefOptionsAddControlDependency(opts, oper)
    ccall((:TF_ImportGraphDefOptionsAddControlDependency, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Ptr{TF_Operation}), opts, oper)
end

function TF_ImportGraphDefOptionsAddReturnOutput(opts, oper_name, index::Cint)
    ccall((:TF_ImportGraphDefOptionsAddReturnOutput, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Ptr{Cchar}, Cint), opts, oper_name, index)
end

function TF_ImportGraphDefOptionsNumReturnOutputs(opts)
    ccall((:TF_ImportGraphDefOptionsNumReturnOutputs, LibTensorFlow), Cint, (Ptr{TF_ImportGraphDefOptions},), opts)
end

function TF_ImportGraphDefOptionsAddReturnOperation(opts, oper_name)
    ccall((:TF_ImportGraphDefOptionsAddReturnOperation, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefOptions}, Ptr{Cchar}), opts, oper_name)
end

function TF_ImportGraphDefOptionsNumReturnOperations(opts)
    ccall((:TF_ImportGraphDefOptionsNumReturnOperations, LibTensorFlow), Cint, (Ptr{TF_ImportGraphDefOptions},), opts)
end

mutable struct TF_ImportGraphDefResults end

function TF_ImportGraphDefResultsReturnOutputs(results, num_outputs, outputs)
    ccall((:TF_ImportGraphDefResultsReturnOutputs, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefResults}, Ptr{Cint}, Ptr{Ptr{TF_Output}}), results, num_outputs, outputs)
end

function TF_ImportGraphDefResultsReturnOperations(results, num_opers, opers)
    ccall((:TF_ImportGraphDefResultsReturnOperations, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefResults}, Ptr{Cint}, Ptr{Ptr{Ptr{TF_Operation}}}), results, num_opers, opers)
end

function TF_ImportGraphDefResultsMissingUnusedInputMappings(results, num_missing_unused_input_mappings, src_names, src_indexes)
    ccall((:TF_ImportGraphDefResultsMissingUnusedInputMappings, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefResults}, Ptr{Cint}, Ptr{Ptr{Ptr{Cchar}}}, Ptr{Ptr{Cint}}), results, num_missing_unused_input_mappings, src_names, src_indexes)
end

function TF_DeleteImportGraphDefResults(results)
    ccall((:TF_DeleteImportGraphDefResults, LibTensorFlow), Cvoid, (Ptr{TF_ImportGraphDefResults},), results)
end

function TF_GraphImportGraphDefWithResults(graph::Ptr{TF_Graph}, graph_def, options, status)
    ccall((:TF_GraphImportGraphDefWithResults, LibTensorFlow), Ptr{TF_ImportGraphDefResults}, (Ptr{TF_Graph}, Ptr{TF_Buffer}, Ptr{TF_ImportGraphDefOptions}, Ptr{TF_Status}), graph, graph_def, options, status)
end

function TF_GraphImportGraphDefWithReturnOutputs(graph::Ptr{TF_Graph}, graph_def, options, return_outputs, num_return_outputs::Cint, status)
    ccall((:TF_GraphImportGraphDefWithReturnOutputs, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Buffer}, Ptr{TF_ImportGraphDefOptions}, Ptr{TF_Output}, Cint, Ptr{TF_Status}), graph, graph_def, options, return_outputs, num_return_outputs, status)
end

function TF_GraphImportGraphDef(graph::Ptr{TF_Graph}, graph_def, options, status)
    ccall((:TF_GraphImportGraphDef, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Buffer}, Ptr{TF_ImportGraphDefOptions}, Ptr{TF_Status}), graph, graph_def, options, status)
end

function TF_GraphCopyFunction(g, func, grad, status)
    ccall((:TF_GraphCopyFunction, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Function}, Ptr{TF_Function}, Ptr{TF_Status}), g, func, grad, status)
end

function TF_GraphNumFunctions(g)
    ccall((:TF_GraphNumFunctions, LibTensorFlow), Cint, (Ptr{TF_Graph},), g)
end

function TF_GraphGetFunctions(g, funcs, max_func::Cint, status)
    ccall((:TF_GraphGetFunctions, LibTensorFlow), Cint, (Ptr{TF_Graph}, Ptr{Ptr{TF_Function}}, Cint, Ptr{TF_Status}), g, funcs, max_func, status)
end

function TF_OperationToNodeDef(oper, output_node_def, status)
    ccall((:TF_OperationToNodeDef, LibTensorFlow), Cvoid, (Ptr{TF_Operation}, Ptr{TF_Buffer}, Ptr{TF_Status}), oper, output_node_def, status)
end

struct TF_WhileParams
    ninputs::Cint
    cond_graph::Ptr{TF_Graph}
    cond_inputs::Ptr{TF_Output}
    cond_output::TF_Output
    body_graph::Ptr{TF_Graph}
    body_inputs::Ptr{TF_Output}
    body_outputs::Ptr{TF_Output}
    name::Ptr{Cchar}
end

function TF_NewWhile(g, inputs, ninputs::Cint, status)
    ccall((:TF_NewWhile, LibTensorFlow), TF_WhileParams, (Ptr{TF_Graph}, Ptr{TF_Output}, Cint, Ptr{TF_Status}), g, inputs, ninputs, status)
end

function TF_FinishWhile(params, status, outputs)
    ccall((:TF_FinishWhile, LibTensorFlow), Cvoid, (Ptr{TF_WhileParams}, Ptr{TF_Status}, Ptr{TF_Output}), params, status, outputs)
end

function TF_AbortWhile(params)
    ccall((:TF_AbortWhile, LibTensorFlow), Cvoid, (Ptr{TF_WhileParams},), params)
end

function TF_AddGradients(g, y, ny::Cint, x, nx::Cint, dx, status, dy)
    ccall((:TF_AddGradients, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Output}, Cint, Ptr{TF_Output}, Cint, Ptr{TF_Output}, Ptr{TF_Status}, Ptr{TF_Output}), g, y, ny, x, nx, dx, status, dy)
end

function TF_AddGradientsWithPrefix(g, prefix, y, ny::Cint, x, nx::Cint, dx, status, dy)
    ccall((:TF_AddGradientsWithPrefix, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{Cchar}, Ptr{TF_Output}, Cint, Ptr{TF_Output}, Cint, Ptr{TF_Output}, Ptr{TF_Status}, Ptr{TF_Output}), g, prefix, y, ny, x, nx, dx, status, dy)
end

function TF_GraphToFunction(fn_body, fn_name, append_hash_to_fn_name::Cuchar, num_opers::Cint, opers, ninputs::Cint, inputs, noutputs::Cint, outputs, output_names, opts, description, status)
    ccall((:TF_GraphToFunction, LibTensorFlow), Ptr{TF_Function}, (Ptr{TF_Graph}, Ptr{Cchar}, Cuchar, Cint, Ptr{Ptr{TF_Operation}}, Cint, Ptr{TF_Output}, Cint, Ptr{TF_Output}, Ptr{Ptr{Cchar}}, Ptr{TF_FunctionOptions}, Ptr{Cchar}, Ptr{TF_Status}), fn_body, fn_name, append_hash_to_fn_name, num_opers, opers, ninputs, inputs, noutputs, outputs, output_names, opts, description, status)
end

function TF_GraphToFunctionWithControlOutputs(fn_body, fn_name, append_hash_to_fn_name::Cuchar, num_opers::Cint, opers, ninputs::Cint, inputs, noutputs::Cint, outputs, output_names, ncontrol_outputs::Cint, control_outputs, control_output_names, opts, description, status)
    ccall((:TF_GraphToFunctionWithControlOutputs, LibTensorFlow), Ptr{TF_Function}, (Ptr{TF_Graph}, Ptr{Cchar}, Cuchar, Cint, Ptr{Ptr{TF_Operation}}, Cint, Ptr{TF_Output}, Cint, Ptr{TF_Output}, Ptr{Ptr{Cchar}}, Cint, Ptr{Ptr{TF_Operation}}, Ptr{Ptr{Cchar}}, Ptr{TF_FunctionOptions}, Ptr{Cchar}, Ptr{TF_Status}), fn_body, fn_name, append_hash_to_fn_name, num_opers, opers, ninputs, inputs, noutputs, outputs, output_names, ncontrol_outputs, control_outputs, control_output_names, opts, description, status)
end

function TF_FunctionName(func)
    ccall((:TF_FunctionName, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_Function},), func)
end

function TF_FunctionToFunctionDef(func, output_func_def, status)
    ccall((:TF_FunctionToFunctionDef, LibTensorFlow), Cvoid, (Ptr{TF_Function}, Ptr{TF_Buffer}, Ptr{TF_Status}), func, output_func_def, status)
end

function TF_FunctionImportFunctionDef(proto, proto_len::Csize_t, status)
    ccall((:TF_FunctionImportFunctionDef, LibTensorFlow), Ptr{TF_Function}, (Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), proto, proto_len, status)
end

function TF_FunctionSetAttrValueProto(func, attr_name, proto, proto_len::Csize_t, status)
    ccall((:TF_FunctionSetAttrValueProto, LibTensorFlow), Cvoid, (Ptr{TF_Function}, Ptr{Cchar}, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), func, attr_name, proto, proto_len, status)
end

function TF_FunctionGetAttrValueProto(func, attr_name, output_attr_value, status)
    ccall((:TF_FunctionGetAttrValueProto, LibTensorFlow), Cvoid, (Ptr{TF_Function}, Ptr{Cchar}, Ptr{TF_Buffer}, Ptr{TF_Status}), func, attr_name, output_attr_value, status)
end

function TF_DeleteFunction(func)
    ccall((:TF_DeleteFunction, LibTensorFlow), Cvoid, (Ptr{TF_Function},), func)
end

function TF_TryEvaluateConstant(graph::Ptr{TF_Graph}, output::TF_Output, result, status)
    ccall((:TF_TryEvaluateConstant, LibTensorFlow), Cuchar, (Ptr{TF_Graph}, TF_Output, Ptr{Ptr{TF_Tensor}}, Ptr{TF_Status}), graph, output, result, status)
end

mutable struct TF_Session end

function TF_NewSession(graph::Ptr{TF_Graph}, opts, status)
    ccall((:TF_NewSession, LibTensorFlow), Ptr{TF_Session}, (Ptr{TF_Graph}, Ptr{TF_SessionOptions}, Ptr{TF_Status}), graph, opts, status)
end

function TF_LoadSessionFromSavedModel(session_options, run_options, export_dir, tags, tags_len::Cint, graph, meta_graph_def, status)
    ccall((:TF_LoadSessionFromSavedModel, LibTensorFlow), Ptr{TF_Session}, (Ptr{TF_SessionOptions}, Ptr{TF_Buffer}, Ptr{Cchar}, Ptr{Ptr{Cchar}}, Cint, Ptr{TF_Graph}, Ptr{TF_Buffer}, Ptr{TF_Status}), session_options, run_options, export_dir, tags, tags_len, graph, meta_graph_def, status)
end

function TF_CloseSession(arg1, status)
    ccall((:TF_CloseSession, LibTensorFlow), Cvoid, (Ptr{TF_Session}, Ptr{TF_Status}), arg1, status)
end

function TF_DeleteSession(arg1, status)
    ccall((:TF_DeleteSession, LibTensorFlow), Cvoid, (Ptr{TF_Session}, Ptr{TF_Status}), arg1, status)
end

function TF_SessionRun(session, run_options, inputs::Vector{TF_Output}, input_values, ninputs::Cint, outputs::Vector{TF_Output}, output_values, noutputs::Cint, target_opers, ntargets::Cint, run_metadata, arg12)
    ccall((:TF_SessionRun, LibTensorFlow), Cvoid, (Ptr{TF_Session}, Ptr{TF_Buffer}, Ptr{TF_Output}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{TF_Output}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{Ptr{TF_Operation}}, Cint, Ptr{TF_Buffer}, Ptr{TF_Status}), session, run_options, inputs, input_values, ninputs, outputs, output_values, noutputs, target_opers, ntargets, run_metadata, arg12)
end

function TF_SessionPRunSetup(arg1, inputs, ninputs::Cint, outputs, noutputs::Cint, target_opers, ntargets::Cint, handle, arg9)
    ccall((:TF_SessionPRunSetup, LibTensorFlow), Cvoid, (Ptr{TF_Session}, Ptr{TF_Output}, Cint, Ptr{TF_Output}, Cint, Ptr{Ptr{TF_Operation}}, Cint, Ptr{Ptr{Cchar}}, Ptr{TF_Status}), arg1, inputs, ninputs, outputs, noutputs, target_opers, ntargets, handle, arg9)
end

function TF_SessionPRun(arg1, handle, inputs, input_values, ninputs::Cint, outputs, output_values, noutputs::Cint, target_opers, ntargets::Cint, arg11)
    ccall((:TF_SessionPRun, LibTensorFlow), Cvoid, (Ptr{TF_Session}, Ptr{Cchar}, Ptr{TF_Output}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{TF_Output}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{Ptr{TF_Operation}}, Cint, Ptr{TF_Status}), arg1, handle, inputs, input_values, ninputs, outputs, output_values, noutputs, target_opers, ntargets, arg11)
end

function TF_DeletePRunHandle(handle)
    ccall((:TF_DeletePRunHandle, LibTensorFlow), Cvoid, (Ptr{Cchar},), handle)
end

mutable struct TF_DeprecatedSession end

function TF_NewDeprecatedSession(arg1, status)
    ccall((:TF_NewDeprecatedSession, LibTensorFlow), Ptr{TF_DeprecatedSession}, (Ptr{TF_SessionOptions}, Ptr{TF_Status}), arg1, status)
end

function TF_CloseDeprecatedSession(arg1, status)
    ccall((:TF_CloseDeprecatedSession, LibTensorFlow), Cvoid, (Ptr{TF_DeprecatedSession}, Ptr{TF_Status}), arg1, status)
end

function TF_DeleteDeprecatedSession(arg1, status)
    ccall((:TF_DeleteDeprecatedSession, LibTensorFlow), Cvoid, (Ptr{TF_DeprecatedSession}, Ptr{TF_Status}), arg1, status)
end

function TF_Reset(opt, containers, ncontainers::Cint, status)
    ccall((:TF_Reset, LibTensorFlow), Cvoid, (Ptr{TF_SessionOptions}, Ptr{Ptr{Cchar}}, Cint, Ptr{TF_Status}), opt, containers, ncontainers, status)
end

function TF_ExtendGraph(arg1, proto, proto_len::Csize_t, arg4)
    ccall((:TF_ExtendGraph, LibTensorFlow), Cvoid, (Ptr{TF_DeprecatedSession}, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), arg1, proto, proto_len, arg4)
end

function TF_Run(arg1, run_options, input_names, inputs, ninputs::Cint, output_names, outputs, noutputs::Cint, target_oper_names, ntargets::Cint, run_metadata, arg12)
    ccall((:TF_Run, LibTensorFlow), Cvoid, (Ptr{TF_DeprecatedSession}, Ptr{TF_Buffer}, Ptr{Ptr{Cchar}}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{Ptr{Cchar}}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{Ptr{Cchar}}, Cint, Ptr{TF_Buffer}, Ptr{TF_Status}), arg1, run_options, input_names, inputs, ninputs, output_names, outputs, noutputs, target_oper_names, ntargets, run_metadata, arg12)
end

function TF_PRunSetup(arg1, input_names, ninputs::Cint, output_names, noutputs::Cint, target_oper_names, ntargets::Cint, handle, arg9)
    ccall((:TF_PRunSetup, LibTensorFlow), Cvoid, (Ptr{TF_DeprecatedSession}, Ptr{Ptr{Cchar}}, Cint, Ptr{Ptr{Cchar}}, Cint, Ptr{Ptr{Cchar}}, Cint, Ptr{Ptr{Cchar}}, Ptr{TF_Status}), arg1, input_names, ninputs, output_names, noutputs, target_oper_names, ntargets, handle, arg9)
end

function TF_PRun(arg1, handle, input_names, inputs, ninputs::Cint, output_names, outputs, noutputs::Cint, target_oper_names, ntargets::Cint, arg11)
    ccall((:TF_PRun, LibTensorFlow), Cvoid, (Ptr{TF_DeprecatedSession}, Ptr{Cchar}, Ptr{Ptr{Cchar}}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{Ptr{Cchar}}, Ptr{Ptr{TF_Tensor}}, Cint, Ptr{Ptr{Cchar}}, Cint, Ptr{TF_Status}), arg1, handle, input_names, inputs, ninputs, output_names, outputs, noutputs, target_oper_names, ntargets, arg11)
end

mutable struct TF_DeviceList end

function TF_SessionListDevices(session, status)
    ccall((:TF_SessionListDevices, LibTensorFlow), Ptr{TF_DeviceList}, (Ptr{TF_Session}, Ptr{TF_Status}), session, status)
end

function TF_DeprecatedSessionListDevices(session, status)
    ccall((:TF_DeprecatedSessionListDevices, LibTensorFlow), Ptr{TF_DeviceList}, (Ptr{TF_DeprecatedSession}, Ptr{TF_Status}), session, status)
end

function TF_DeleteDeviceList(list)
    ccall((:TF_DeleteDeviceList, LibTensorFlow), Cvoid, (Ptr{TF_DeviceList},), list)
end

function TF_DeviceListCount(list)
    ccall((:TF_DeviceListCount, LibTensorFlow), Cint, (Ptr{TF_DeviceList},), list)
end

function TF_DeviceListName(list, index::Cint, status)
    ccall((:TF_DeviceListName, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_DeviceList}, Cint, Ptr{TF_Status}), list, index, status)
end

function TF_DeviceListType(list, index::Cint, status)
    ccall((:TF_DeviceListType, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_DeviceList}, Cint, Ptr{TF_Status}), list, index, status)
end

function TF_DeviceListMemoryBytes(list, index::Cint, status)
    ccall((:TF_DeviceListMemoryBytes, LibTensorFlow), Int64, (Ptr{TF_DeviceList}, Cint, Ptr{TF_Status}), list, index, status)
end

function TF_DeviceListIncarnation(list, index::Cint, status)
    ccall((:TF_DeviceListIncarnation, LibTensorFlow), UInt64, (Ptr{TF_DeviceList}, Cint, Ptr{TF_Status}), list, index, status)
end

mutable struct TF_Library end

function TF_LoadLibrary(library_filename, status)
    ccall((:TF_LoadLibrary, LibTensorFlow), Ptr{TF_Library}, (Ptr{Cchar}, Ptr{TF_Status}), library_filename, status)
end

function TF_GetOpList(lib_handle)
    ccall((:TF_GetOpList, LibTensorFlow), TF_Buffer, (Ptr{TF_Library},), lib_handle)
end

function TF_DeleteLibraryHandle(lib_handle)
    ccall((:TF_DeleteLibraryHandle, LibTensorFlow), Cvoid, (Ptr{TF_Library},), lib_handle)
end

function TF_GetAllOpList()
    ccall((:TF_GetAllOpList, LibTensorFlow), Ptr{TF_Buffer}, ())
end

mutable struct TF_ApiDefMap end

function TF_NewApiDefMap(op_list_buffer, status)
    ccall((:TF_NewApiDefMap, LibTensorFlow), Ptr{TF_ApiDefMap}, (Ptr{TF_Buffer}, Ptr{TF_Status}), op_list_buffer, status)
end

function TF_DeleteApiDefMap(apimap)
    ccall((:TF_DeleteApiDefMap, LibTensorFlow), Cvoid, (Ptr{TF_ApiDefMap},), apimap)
end

function TF_ApiDefMapPut(api_def_map, text, text_len::Csize_t, status)
    ccall((:TF_ApiDefMapPut, LibTensorFlow), Cvoid, (Ptr{TF_ApiDefMap}, Ptr{Cchar}, Csize_t, Ptr{TF_Status}), api_def_map, text, text_len, status)
end

function TF_ApiDefMapGet(api_def_map, name, name_len::Csize_t, status)
    ccall((:TF_ApiDefMapGet, LibTensorFlow), Ptr{TF_Buffer}, (Ptr{TF_ApiDefMap}, Ptr{Cchar}, Csize_t, Ptr{TF_Status}), api_def_map, name, name_len, status)
end

function TF_GetAllRegisteredKernels(status)
    ccall((:TF_GetAllRegisteredKernels, LibTensorFlow), Ptr{TF_Buffer}, (Ptr{TF_Status},), status)
end

function TF_GetRegisteredKernelsForOp(name, status)
    ccall((:TF_GetRegisteredKernelsForOp, LibTensorFlow), Ptr{TF_Buffer}, (Ptr{Cchar}, Ptr{TF_Status}), name, status)
end

function TF_UpdateEdge(graph::Ptr{TF_Graph}, new_src::TF_Output, dst::TF_Input, status)
    ccall((:TF_UpdateEdge, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, TF_Output, TF_Input, Ptr{TF_Status}), graph, new_src, dst, status)
end

mutable struct TF_Server end

function TF_NewServer(proto, proto_len::Csize_t, status)
    ccall((:TF_NewServer, LibTensorFlow), Ptr{TF_Server}, (Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), proto, proto_len, status)
end

function TF_ServerStart(server, status)
    ccall((:TF_ServerStart, LibTensorFlow), Cvoid, (Ptr{TF_Server}, Ptr{TF_Status}), server, status)
end

function TF_ServerStop(server, status)
    ccall((:TF_ServerStop, LibTensorFlow), Cvoid, (Ptr{TF_Server}, Ptr{TF_Status}), server, status)
end

function TF_ServerJoin(server, status)
    ccall((:TF_ServerJoin, LibTensorFlow), Cvoid, (Ptr{TF_Server}, Ptr{TF_Status}), server, status)
end

function TF_ServerTarget(server)
    ccall((:TF_ServerTarget, LibTensorFlow), Ptr{Cchar}, (Ptr{TF_Server},), server)
end

function TF_DeleteServer(server)
    ccall((:TF_DeleteServer, LibTensorFlow), Cvoid, (Ptr{TF_Server},), server)
end

function TF_RegisterLogListener(listener)
    ccall((:TF_RegisterLogListener, LibTensorFlow), Cvoid, (Ptr{Cvoid},), listener)
end

function TF_RegisterFilesystemPlugin(plugin_filename, status)
    ccall((:TF_RegisterFilesystemPlugin, LibTensorFlow), Cvoid, (Ptr{Cchar}, Ptr{TF_Status}), plugin_filename, status)
end

function TF_AddOperationControlInput(graph::Ptr{TF_Graph}, op, input)
    ccall((:TF_AddOperationControlInput, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Operation}, Ptr{TF_Operation}), graph, op, input)
end

function TF_SetAttr(graph::Ptr{TF_Graph}, op, attr_name, attr_value_proto, status)
    ccall((:TF_SetAttr, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_Buffer}, Ptr{TF_Status}), graph, op, attr_name, attr_value_proto, status)
end

function TF_ClearAttr(graph::Ptr{TF_Graph}, op, attr_name, status)
    ccall((:TF_ClearAttr, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Operation}, Ptr{Cchar}, Ptr{TF_Status}), graph, op, attr_name, status)
end

function TF_SetFullType(graph::Ptr{TF_Graph}, op, full_type_proto)
    ccall((:TF_SetFullType, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Operation}, Ptr{TF_Buffer}), graph, op, full_type_proto)
end

function TF_SetRequestedDevice(graph::Ptr{TF_Graph}, op, device)
    ccall((:TF_SetRequestedDevice, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Operation}, Ptr{Cchar}), graph, op, device)
end

function TF_RemoveAllControlInputs(graph::Ptr{TF_Graph}, op)
    ccall((:TF_RemoveAllControlInputs, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Ptr{TF_Operation}), graph, op)
end

function TF_SetRequireShapeInferenceFns(graph::Ptr{TF_Graph}, require::Bool)
    ccall((:TF_SetRequireShapeInferenceFns, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, Bool), graph, require)
end

function TF_ExtendSession(session, status)
    ccall((:TF_ExtendSession, LibTensorFlow), Cvoid, (Ptr{TF_Session}, Ptr{TF_Status}), session, status)
end

function TF_GetHandleShapeAndType(graph::Ptr{TF_Graph}, output::TF_Output)
    ccall((:TF_GetHandleShapeAndType, LibTensorFlow), Ptr{TF_Buffer}, (Ptr{TF_Graph}, TF_Output), graph, output)
end

function TF_SetHandleShapeAndType(graph::Ptr{TF_Graph}, output::TF_Output, proto, proto_len::Csize_t, status)
    ccall((:TF_SetHandleShapeAndType, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, TF_Output, Ptr{Cvoid}, Csize_t, Ptr{TF_Status}), graph, output, proto, proto_len, status)
end

function TF_AddWhileInputHack(graph::Ptr{TF_Graph}, new_src::TF_Output, dst, status)
    ccall((:TF_AddWhileInputHack, LibTensorFlow), Cvoid, (Ptr{TF_Graph}, TF_Output, Ptr{TF_Operation}, Ptr{TF_Status}), graph, new_src, dst, status)
end


const TF_Bool = Cuchar

const TF_OK = TSL_OK

const TF_CANCELLED = TSL_CANCELLED

const TF_UNKNOWN = TSL_UNKNOWN

const TF_INVALID_ARGUMENT = TSL_INVALID_ARGUMENT

const TF_DEADLINE_EXCEEDED = TSL_DEADLINE_EXCEEDED

const TF_NOT_FOUND = TSL_NOT_FOUND

const TF_ALREADY_EXISTS = TSL_ALREADY_EXISTS

const TF_PERMISSION_DENIED = TSL_PERMISSION_DENIED

const TF_UNAUTHENTICATED = TSL_UNAUTHENTICATED

const TF_RESOURCE_EXHAUSTED = TSL_RESOURCE_EXHAUSTED

const TF_FAILED_PRECONDITION = TSL_FAILED_PRECONDITION

const TF_ABORTED = TSL_ABORTED

const TF_OUT_OF_RANGE = TSL_OUT_OF_RANGE

const TF_UNIMPLEMENTED = TSL_UNIMPLEMENTED

const TF_INTERNAL = TSL_INTERNAL

const TF_UNAVAILABLE = TSL_UNAVAILABLE

const TF_DATA_LOSS = TSL_DATA_LOSS

const TF_PayloadVisitor = TSL_PayloadVisitor

const TF_TSTRING_LITTLE_ENDIAN = 1

# exports
const PREFIXES = ["TF"]
for name in names(@__MODULE__; all=true), prefix in PREFIXES
    if startswith(string(name), prefix)
        @eval export $name
    end
end


