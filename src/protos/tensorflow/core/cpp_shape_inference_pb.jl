# Autogenerated using ProtoBuf.jl v1.0.15 on 2024-03-03T12:31:43.965

import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export CppShapeInferenceInputsNeeded, var"CppShapeInferenceResult.HandleShapeAndType"
export var"CppShapeInferenceResult.HandleData", CppShapeInferenceResult

struct CppShapeInferenceInputsNeeded
    input_tensors_needed::Vector{Int32}
    input_tensors_as_shapes_needed::Vector{Int32}
end
PB.default_values(::Type{CppShapeInferenceInputsNeeded}) = (;input_tensors_needed = Vector{Int32}(), input_tensors_as_shapes_needed = Vector{Int32}())
PB.field_numbers(::Type{CppShapeInferenceInputsNeeded}) = (;input_tensors_needed = 1, input_tensors_as_shapes_needed = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CppShapeInferenceInputsNeeded})
    input_tensors_needed = PB.BufferedVector{Int32}()
    input_tensors_as_shapes_needed = PB.BufferedVector{Int32}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, wire_type, input_tensors_needed)
        elseif field_number == 2
            PB.decode!(d, wire_type, input_tensors_as_shapes_needed)
        else
            PB.skip(d, wire_type)
        end
    end
    return CppShapeInferenceInputsNeeded(input_tensors_needed[], input_tensors_as_shapes_needed[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CppShapeInferenceInputsNeeded)
    initpos = position(e.io)
    !isempty(x.input_tensors_needed) && PB.encode(e, 1, x.input_tensors_needed)
    !isempty(x.input_tensors_as_shapes_needed) && PB.encode(e, 2, x.input_tensors_as_shapes_needed)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CppShapeInferenceInputsNeeded)
    encoded_size = 0
    !isempty(x.input_tensors_needed) && (encoded_size += PB._encoded_size(x.input_tensors_needed, 1))
    !isempty(x.input_tensors_as_shapes_needed) && (encoded_size += PB._encoded_size(x.input_tensors_as_shapes_needed, 2))
    return encoded_size
end

struct var"CppShapeInferenceResult.HandleShapeAndType"
    shape::Union{Nothing,tensorflow.TensorShapeProto}
    dtype::tensorflow.var"#DataType".T
    var"#type"::Union{Nothing,tensorflow.FullTypeDef}
end
PB.reserved_fields(::Type{var"CppShapeInferenceResult.HandleShapeAndType"}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[3])
PB.default_values(::Type{var"CppShapeInferenceResult.HandleShapeAndType"}) = (;shape = nothing, dtype = tensorflow.var"#DataType".DT_INVALID, var"#type" = nothing)
PB.field_numbers(::Type{var"CppShapeInferenceResult.HandleShapeAndType"}) = (;shape = 1, dtype = 2, var"#type" = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"CppShapeInferenceResult.HandleShapeAndType"})
    shape = Ref{Union{Nothing,tensorflow.TensorShapeProto}}(nothing)
    dtype = tensorflow.var"#DataType".DT_INVALID
    var"#type" = Ref{Union{Nothing,tensorflow.FullTypeDef}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, shape)
        elseif field_number == 2
            dtype = PB.decode(d, tensorflow.var"#DataType".T)
        elseif field_number == 4
            PB.decode!(d, var"#type")
        else
            PB.skip(d, wire_type)
        end
    end
    return var"CppShapeInferenceResult.HandleShapeAndType"(shape[], dtype, var"#type"[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"CppShapeInferenceResult.HandleShapeAndType")
    initpos = position(e.io)
    !isnothing(x.shape) && PB.encode(e, 1, x.shape)
    x.dtype != tensorflow.var"#DataType".DT_INVALID && PB.encode(e, 2, x.dtype)
    !isnothing(x.var"#type") && PB.encode(e, 4, x.var"#type")
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"CppShapeInferenceResult.HandleShapeAndType")
    encoded_size = 0
    !isnothing(x.shape) && (encoded_size += PB._encoded_size(x.shape, 1))
    x.dtype != tensorflow.var"#DataType".DT_INVALID && (encoded_size += PB._encoded_size(x.dtype, 2))
    !isnothing(x.var"#type") && (encoded_size += PB._encoded_size(x.var"#type", 4))
    return encoded_size
end

struct var"CppShapeInferenceResult.HandleData"
    is_set::Bool
    shape_and_type::Vector{var"CppShapeInferenceResult.HandleShapeAndType"}
end
PB.default_values(::Type{var"CppShapeInferenceResult.HandleData"}) = (;is_set = false, shape_and_type = Vector{var"CppShapeInferenceResult.HandleShapeAndType"}())
PB.field_numbers(::Type{var"CppShapeInferenceResult.HandleData"}) = (;is_set = 1, shape_and_type = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"CppShapeInferenceResult.HandleData"})
    is_set = false
    shape_and_type = PB.BufferedVector{var"CppShapeInferenceResult.HandleShapeAndType"}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            is_set = PB.decode(d, Bool)
        elseif field_number == 2
            PB.decode!(d, shape_and_type)
        else
            PB.skip(d, wire_type)
        end
    end
    return var"CppShapeInferenceResult.HandleData"(is_set, shape_and_type[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"CppShapeInferenceResult.HandleData")
    initpos = position(e.io)
    x.is_set != false && PB.encode(e, 1, x.is_set)
    !isempty(x.shape_and_type) && PB.encode(e, 2, x.shape_and_type)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"CppShapeInferenceResult.HandleData")
    encoded_size = 0
    x.is_set != false && (encoded_size += PB._encoded_size(x.is_set, 1))
    !isempty(x.shape_and_type) && (encoded_size += PB._encoded_size(x.shape_and_type, 2))
    return encoded_size
end

struct CppShapeInferenceResult
    shape::Union{Nothing,tensorflow.TensorShapeProto}
    handle_data::Union{Nothing,var"CppShapeInferenceResult.HandleData"}
end
PB.reserved_fields(::Type{CppShapeInferenceResult}) = (names = String[], numbers = Union{Int,UnitRange{Int}}[2, 3])
PB.default_values(::Type{CppShapeInferenceResult}) = (;shape = nothing, handle_data = nothing)
PB.field_numbers(::Type{CppShapeInferenceResult}) = (;shape = 1, handle_data = 4)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:CppShapeInferenceResult})
    shape = Ref{Union{Nothing,tensorflow.TensorShapeProto}}(nothing)
    handle_data = Ref{Union{Nothing,var"CppShapeInferenceResult.HandleData"}}(nothing)
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            PB.decode!(d, shape)
        elseif field_number == 4
            PB.decode!(d, handle_data)
        else
            PB.skip(d, wire_type)
        end
    end
    return CppShapeInferenceResult(shape[], handle_data[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::CppShapeInferenceResult)
    initpos = position(e.io)
    !isnothing(x.shape) && PB.encode(e, 1, x.shape)
    !isnothing(x.handle_data) && PB.encode(e, 4, x.handle_data)
    return position(e.io) - initpos
end
function PB._encoded_size(x::CppShapeInferenceResult)
    encoded_size = 0
    !isnothing(x.shape) && (encoded_size += PB._encoded_size(x.shape, 1))
    !isnothing(x.handle_data) && (encoded_size += PB._encoded_size(x.handle_data, 4))
    return encoded_size
end
