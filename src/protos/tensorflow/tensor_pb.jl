# Autogenerated using ProtoBuf.jl v1.0.15 on 2024-03-03T12:31:44.075

import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export TensorProto, VariantTensorDataProto

# Abstract types to help resolve mutually recursive definitions
abstract type var"##AbstractTensorProto" end
abstract type var"##AbstractVariantTensorDataProto" end


struct TensorProto{T1<:Union{Nothing,var"##AbstractVariantTensorDataProto"}} <: var"##AbstractTensorProto"
    dtype::var"#DataType".T
    tensor_shape::Union{Nothing,TensorShapeProto}
    version_number::Int32
    tensor_content::Vector{UInt8}
    half_val::Vector{Int32}
    float_val::Vector{Float32}
    double_val::Vector{Float64}
    int_val::Vector{Int32}
    string_val::Vector{Vector{UInt8}}
    scomplex_val::Vector{Float32}
    int64_val::Vector{Int64}
    bool_val::Vector{Bool}
    dcomplex_val::Vector{Float64}
    resource_handle_val::Vector{ResourceHandleProto}
    variant_val::Vector{T1}
    uint32_val::Vector{UInt32}
    uint64_val::Vector{UInt64}
    float8_val::Vector{UInt8}
end
PB.default_values(::Type{TensorProto}) = (;dtype = var"#DataType".DT_INVALID, tensor_shape = nothing, version_number = zero(Int32), tensor_content = UInt8[], half_val = Vector{Int32}(), float_val = Vector{Float32}(), double_val = Vector{Float64}(), int_val = Vector{Int32}(), string_val = Vector{Vector{UInt8}}(), scomplex_val = Vector{Float32}(), int64_val = Vector{Int64}(), bool_val = Vector{Bool}(), dcomplex_val = Vector{Float64}(), resource_handle_val = Vector{ResourceHandleProto}(), variant_val = Vector{VariantTensorDataProto}(), uint32_val = Vector{UInt32}(), uint64_val = Vector{UInt64}(), float8_val = UInt8[])
PB.field_numbers(::Type{TensorProto}) = (;dtype = 1, tensor_shape = 2, version_number = 3, tensor_content = 4, half_val = 13, float_val = 5, double_val = 6, int_val = 7, string_val = 8, scomplex_val = 9, int64_val = 10, bool_val = 11, dcomplex_val = 12, resource_handle_val = 14, variant_val = 15, uint32_val = 16, uint64_val = 17, float8_val = 18)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:TensorProto})
    dtype = var"#DataType".DT_INVALID
    tensor_shape = Ref{Union{Nothing,TensorShapeProto}}(nothing)
    version_number = zero(Int32)
    tensor_content = UInt8[]
    half_val = PB.BufferedVector{Int32}()
    float_val = PB.BufferedVector{Float32}()
    double_val = PB.BufferedVector{Float64}()
    int_val = PB.BufferedVector{Int32}()
    string_val = PB.BufferedVector{Vector{UInt8}}()
    scomplex_val = PB.BufferedVector{Float32}()
    int64_val = PB.BufferedVector{Int64}()
    bool_val = PB.BufferedVector{Bool}()
    dcomplex_val = PB.BufferedVector{Float64}()
    resource_handle_val = PB.BufferedVector{ResourceHandleProto}()
    variant_val = PB.BufferedVector{VariantTensorDataProto}()
    uint32_val = PB.BufferedVector{UInt32}()
    uint64_val = PB.BufferedVector{UInt64}()
    float8_val = UInt8[]
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            dtype = PB.decode(d, var"#DataType".T)
        elseif field_number == 2
            PB.decode!(d, tensor_shape)
        elseif field_number == 3
            version_number = PB.decode(d, Int32)
        elseif field_number == 4
            tensor_content = PB.decode(d, Vector{UInt8})
        elseif field_number == 13
            PB.decode!(d, wire_type, half_val)
        elseif field_number == 5
            PB.decode!(d, wire_type, float_val)
        elseif field_number == 6
            PB.decode!(d, wire_type, double_val)
        elseif field_number == 7
            PB.decode!(d, wire_type, int_val)
        elseif field_number == 8
            PB.decode!(d, string_val)
        elseif field_number == 9
            PB.decode!(d, wire_type, scomplex_val)
        elseif field_number == 10
            PB.decode!(d, wire_type, int64_val)
        elseif field_number == 11
            PB.decode!(d, wire_type, bool_val)
        elseif field_number == 12
            PB.decode!(d, wire_type, dcomplex_val)
        elseif field_number == 14
            PB.decode!(d, resource_handle_val)
        elseif field_number == 15
            PB.decode!(d, variant_val)
        elseif field_number == 16
            PB.decode!(d, wire_type, uint32_val)
        elseif field_number == 17
            PB.decode!(d, wire_type, uint64_val)
        elseif field_number == 18
            float8_val = PB.decode(d, Vector{UInt8})
        else
            PB.skip(d, wire_type)
        end
    end
    return TensorProto(dtype, tensor_shape[], version_number, tensor_content, half_val[], float_val[], double_val[], int_val[], string_val[], scomplex_val[], int64_val[], bool_val[], dcomplex_val[], resource_handle_val[], variant_val[], uint32_val[], uint64_val[], float8_val)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::TensorProto)
    initpos = position(e.io)
    x.dtype != var"#DataType".DT_INVALID && PB.encode(e, 1, x.dtype)
    !isnothing(x.tensor_shape) && PB.encode(e, 2, x.tensor_shape)
    x.version_number != zero(Int32) && PB.encode(e, 3, x.version_number)
    !isempty(x.tensor_content) && PB.encode(e, 4, x.tensor_content)
    !isempty(x.half_val) && PB.encode(e, 13, x.half_val)
    !isempty(x.float_val) && PB.encode(e, 5, x.float_val)
    !isempty(x.double_val) && PB.encode(e, 6, x.double_val)
    !isempty(x.int_val) && PB.encode(e, 7, x.int_val)
    !isempty(x.string_val) && PB.encode(e, 8, x.string_val)
    !isempty(x.scomplex_val) && PB.encode(e, 9, x.scomplex_val)
    !isempty(x.int64_val) && PB.encode(e, 10, x.int64_val)
    !isempty(x.bool_val) && PB.encode(e, 11, x.bool_val)
    !isempty(x.dcomplex_val) && PB.encode(e, 12, x.dcomplex_val)
    !isempty(x.resource_handle_val) && PB.encode(e, 14, x.resource_handle_val)
    !isempty(x.variant_val) && PB.encode(e, 15, x.variant_val)
    !isempty(x.uint32_val) && PB.encode(e, 16, x.uint32_val)
    !isempty(x.uint64_val) && PB.encode(e, 17, x.uint64_val)
    !isempty(x.float8_val) && PB.encode(e, 18, x.float8_val)
    return position(e.io) - initpos
end
function PB._encoded_size(x::TensorProto)
    encoded_size = 0
    x.dtype != var"#DataType".DT_INVALID && (encoded_size += PB._encoded_size(x.dtype, 1))
    !isnothing(x.tensor_shape) && (encoded_size += PB._encoded_size(x.tensor_shape, 2))
    x.version_number != zero(Int32) && (encoded_size += PB._encoded_size(x.version_number, 3))
    !isempty(x.tensor_content) && (encoded_size += PB._encoded_size(x.tensor_content, 4))
    !isempty(x.half_val) && (encoded_size += PB._encoded_size(x.half_val, 13))
    !isempty(x.float_val) && (encoded_size += PB._encoded_size(x.float_val, 5))
    !isempty(x.double_val) && (encoded_size += PB._encoded_size(x.double_val, 6))
    !isempty(x.int_val) && (encoded_size += PB._encoded_size(x.int_val, 7))
    !isempty(x.string_val) && (encoded_size += PB._encoded_size(x.string_val, 8))
    !isempty(x.scomplex_val) && (encoded_size += PB._encoded_size(x.scomplex_val, 9))
    !isempty(x.int64_val) && (encoded_size += PB._encoded_size(x.int64_val, 10))
    !isempty(x.bool_val) && (encoded_size += PB._encoded_size(x.bool_val, 11))
    !isempty(x.dcomplex_val) && (encoded_size += PB._encoded_size(x.dcomplex_val, 12))
    !isempty(x.resource_handle_val) && (encoded_size += PB._encoded_size(x.resource_handle_val, 14))
    !isempty(x.variant_val) && (encoded_size += PB._encoded_size(x.variant_val, 15))
    !isempty(x.uint32_val) && (encoded_size += PB._encoded_size(x.uint32_val, 16))
    !isempty(x.uint64_val) && (encoded_size += PB._encoded_size(x.uint64_val, 17))
    !isempty(x.float8_val) && (encoded_size += PB._encoded_size(x.float8_val, 18))
    return encoded_size
end

struct VariantTensorDataProto <: var"##AbstractVariantTensorDataProto"
    type_name::String
    metadata::Vector{UInt8}
    tensors::Vector{<:TensorProto}
end
PB.default_values(::Type{VariantTensorDataProto}) = (;type_name = "", metadata = UInt8[], tensors = Vector{TensorProto}())
PB.field_numbers(::Type{VariantTensorDataProto}) = (;type_name = 1, metadata = 2, tensors = 3)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:VariantTensorDataProto})
    type_name = ""
    metadata = UInt8[]
    tensors = PB.BufferedVector{TensorProto}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            type_name = PB.decode(d, String)
        elseif field_number == 2
            metadata = PB.decode(d, Vector{UInt8})
        elseif field_number == 3
            PB.decode!(d, tensors)
        else
            PB.skip(d, wire_type)
        end
    end
    return VariantTensorDataProto(type_name, metadata, tensors[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::VariantTensorDataProto)
    initpos = position(e.io)
    !isempty(x.type_name) && PB.encode(e, 1, x.type_name)
    !isempty(x.metadata) && PB.encode(e, 2, x.metadata)
    !isempty(x.tensors) && PB.encode(e, 3, x.tensors)
    return position(e.io) - initpos
end
function PB._encoded_size(x::VariantTensorDataProto)
    encoded_size = 0
    !isempty(x.type_name) && (encoded_size += PB._encoded_size(x.type_name, 1))
    !isempty(x.metadata) && (encoded_size += PB._encoded_size(x.metadata, 2))
    !isempty(x.tensors) && (encoded_size += PB._encoded_size(x.tensors, 3))
    return encoded_size
end
