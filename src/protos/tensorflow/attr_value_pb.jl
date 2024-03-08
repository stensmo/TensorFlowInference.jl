# Autogenerated using ProtoBuf.jl v1.0.15 on 2024-03-03T12:31:44.059

import ProtoBuf as PB
using ProtoBuf: OneOf
using ProtoBuf.EnumX: @enumx

export NameAttrList, var"AttrValue.ListValue", AttrValue

# Abstract types to help resolve mutually recursive definitions
abstract type var"##AbstractNameAttrList" end
abstract type var"##AbstractAttrValue.ListValue" end
abstract type var"##AbstractAttrValue" end


struct NameAttrList{T1<:Union{Nothing,var"##AbstractAttrValue"}} <: var"##AbstractNameAttrList"
    name::String
    attr::Dict{String,T1}
end
PB.default_values(::Type{NameAttrList}) = (;name = "", attr = Dict{String,AttrValue}())
PB.field_numbers(::Type{NameAttrList}) = (;name = 1, attr = 2)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:NameAttrList})
    name = ""
    attr = Dict{String,AttrValue}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 1
            name = PB.decode(d, String)
        elseif field_number == 2
            PB.decode!(d, attr)
        else
            PB.skip(d, wire_type)
        end
    end
    return NameAttrList(name, attr)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::NameAttrList)
    initpos = position(e.io)
    !isempty(x.name) && PB.encode(e, 1, x.name)
    !isempty(x.attr) && PB.encode(e, 2, x.attr)
    return position(e.io) - initpos
end
function PB._encoded_size(x::NameAttrList)
    encoded_size = 0
    !isempty(x.name) && (encoded_size += PB._encoded_size(x.name, 1))
    !isempty(x.attr) && (encoded_size += PB._encoded_size(x.attr, 2))
    return encoded_size
end

struct var"AttrValue.ListValue" <: var"##AbstractAttrValue.ListValue"
    s::Vector{Vector{UInt8}}
    i::Vector{Int64}
    f::Vector{Float32}
    b::Vector{Bool}
    var"#type"::Vector{var"#DataType".T}
    shape::Vector{TensorShapeProto}
    tensor::Vector{TensorProto}
    func::Vector{<:NameAttrList}
end
PB.default_values(::Type{var"AttrValue.ListValue"}) = (;s = Vector{Vector{UInt8}}(), i = Vector{Int64}(), f = Vector{Float32}(), b = Vector{Bool}(), var"#type" = Vector{var"#DataType".T}(), shape = Vector{TensorShapeProto}(), tensor = Vector{TensorProto}(), func = Vector{NameAttrList}())
PB.field_numbers(::Type{var"AttrValue.ListValue"}) = (;s = 2, i = 3, f = 4, b = 5, var"#type" = 6, shape = 7, tensor = 8, func = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:var"AttrValue.ListValue"})
    s = PB.BufferedVector{Vector{UInt8}}()
    i = PB.BufferedVector{Int64}()
    f = PB.BufferedVector{Float32}()
    b = PB.BufferedVector{Bool}()
    var"#type" = PB.BufferedVector{var"#DataType".T}()
    shape = PB.BufferedVector{TensorShapeProto}()
    tensor = PB.BufferedVector{TensorProto}()
    func = PB.BufferedVector{NameAttrList}()
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2
            PB.decode!(d, s)
        elseif field_number == 3
            PB.decode!(d, wire_type, i)
        elseif field_number == 4
            PB.decode!(d, wire_type, f)
        elseif field_number == 5
            PB.decode!(d, wire_type, b)
        elseif field_number == 6
            PB.decode!(d, wire_type, var"#type")
        elseif field_number == 7
            PB.decode!(d, shape)
        elseif field_number == 8
            PB.decode!(d, tensor)
        elseif field_number == 9
            PB.decode!(d, func)
        else
            PB.skip(d, wire_type)
        end
    end
    return var"AttrValue.ListValue"(s[], i[], f[], b[], var"#type"[], shape[], tensor[], func[])
end

function PB.encode(e::PB.AbstractProtoEncoder, x::var"AttrValue.ListValue")
    initpos = position(e.io)
    !isempty(x.s) && PB.encode(e, 2, x.s)
    !isempty(x.i) && PB.encode(e, 3, x.i)
    !isempty(x.f) && PB.encode(e, 4, x.f)
    !isempty(x.b) && PB.encode(e, 5, x.b)
    !isempty(x.var"#type") && PB.encode(e, 6, x.var"#type")
    !isempty(x.shape) && PB.encode(e, 7, x.shape)
    !isempty(x.tensor) && PB.encode(e, 8, x.tensor)
    !isempty(x.func) && PB.encode(e, 9, x.func)
    return position(e.io) - initpos
end
function PB._encoded_size(x::var"AttrValue.ListValue")
    encoded_size = 0
    !isempty(x.s) && (encoded_size += PB._encoded_size(x.s, 2))
    !isempty(x.i) && (encoded_size += PB._encoded_size(x.i, 3))
    !isempty(x.f) && (encoded_size += PB._encoded_size(x.f, 4))
    !isempty(x.b) && (encoded_size += PB._encoded_size(x.b, 5))
    !isempty(x.var"#type") && (encoded_size += PB._encoded_size(x.var"#type", 6))
    !isempty(x.shape) && (encoded_size += PB._encoded_size(x.shape, 7))
    !isempty(x.tensor) && (encoded_size += PB._encoded_size(x.tensor, 8))
    !isempty(x.func) && (encoded_size += PB._encoded_size(x.func, 9))
    return encoded_size
end

struct AttrValue <: var"##AbstractAttrValue"
    value::Union{Nothing,OneOf{<:Union{Vector{UInt8},Int64,Float32,Bool,var"#DataType".T,TensorShapeProto,TensorProto,var"##AbstractAttrValue.ListValue",var"##AbstractNameAttrList",String}}}
end
PB.oneof_field_types(::Type{AttrValue}) = (;
    value = (;s=Vector{UInt8}, i=Int64, f=Float32, b=Bool, var"#type"=var"#DataType".T, shape=TensorShapeProto, tensor=TensorProto, list=var"AttrValue.ListValue", func=NameAttrList, placeholder=String),
)
PB.default_values(::Type{AttrValue}) = (;s = UInt8[], i = zero(Int64), f = zero(Float32), b = false, var"#type" = var"#DataType".DT_INVALID, shape = nothing, tensor = nothing, list = nothing, func = nothing, placeholder = "")
PB.field_numbers(::Type{AttrValue}) = (;s = 2, i = 3, f = 4, b = 5, var"#type" = 6, shape = 7, tensor = 8, list = 1, func = 10, placeholder = 9)

function PB.decode(d::PB.AbstractProtoDecoder, ::Type{<:AttrValue})
    value = nothing
    while !PB.message_done(d)
        field_number, wire_type = PB.decode_tag(d)
        if field_number == 2
            value = OneOf(:s, PB.decode(d, Vector{UInt8}))
        elseif field_number == 3
            value = OneOf(:i, PB.decode(d, Int64))
        elseif field_number == 4
            value = OneOf(:f, PB.decode(d, Float32))
        elseif field_number == 5
            value = OneOf(:b, PB.decode(d, Bool))
        elseif field_number == 6
            value = OneOf(:var"#type", PB.decode(d, var"#DataType".T))
        elseif field_number == 7
            value = OneOf(:shape, PB.decode(d, Ref{TensorShapeProto}))
        elseif field_number == 8
            value = OneOf(:tensor, PB.decode(d, Ref{TensorProto}))
        elseif field_number == 1
            value = OneOf(:list, PB.decode(d, Ref{var"AttrValue.ListValue"}))
        elseif field_number == 10
            value = OneOf(:func, PB.decode(d, Ref{NameAttrList}))
        elseif field_number == 9
            value = OneOf(:placeholder, PB.decode(d, String))
        else
            PB.skip(d, wire_type)
        end
    end
    return AttrValue(value)
end

function PB.encode(e::PB.AbstractProtoEncoder, x::AttrValue)
    initpos = position(e.io)
    if isnothing(x.value);
    elseif x.value.name === :s
        PB.encode(e, 2, x.value[]::Vector{UInt8})
    elseif x.value.name === :i
        PB.encode(e, 3, x.value[]::Int64)
    elseif x.value.name === :f
        PB.encode(e, 4, x.value[]::Float32)
    elseif x.value.name === :b
        PB.encode(e, 5, x.value[]::Bool)
    elseif x.value.name === :var"#type"
        PB.encode(e, 6, x.value[]::var"#DataType".T)
    elseif x.value.name === :shape
        PB.encode(e, 7, x.value[]::TensorShapeProto)
    elseif x.value.name === :tensor
        PB.encode(e, 8, x.value[]::TensorProto)
    elseif x.value.name === :list
        PB.encode(e, 1, x.value[]::var"AttrValue.ListValue")
    elseif x.value.name === :func
        PB.encode(e, 10, x.value[]::NameAttrList)
    elseif x.value.name === :placeholder
        PB.encode(e, 9, x.value[]::String)
    end
    return position(e.io) - initpos
end
function PB._encoded_size(x::AttrValue)
    encoded_size = 0
    if isnothing(x.value);
    elseif x.value.name === :s
        encoded_size += PB._encoded_size(x.value[]::Vector{UInt8}, 2)
    elseif x.value.name === :i
        encoded_size += PB._encoded_size(x.value[]::Int64, 3)
    elseif x.value.name === :f
        encoded_size += PB._encoded_size(x.value[]::Float32, 4)
    elseif x.value.name === :b
        encoded_size += PB._encoded_size(x.value[]::Bool, 5)
    elseif x.value.name === :var"#type"
        encoded_size += PB._encoded_size(x.value[]::var"#DataType".T, 6)
    elseif x.value.name === :shape
        encoded_size += PB._encoded_size(x.value[]::TensorShapeProto, 7)
    elseif x.value.name === :tensor
        encoded_size += PB._encoded_size(x.value[]::TensorProto, 8)
    elseif x.value.name === :list
        encoded_size += PB._encoded_size(x.value[]::var"AttrValue.ListValue", 1)
    elseif x.value.name === :func
        encoded_size += PB._encoded_size(x.value[]::NameAttrList, 10)
    elseif x.value.name === :placeholder
        encoded_size += PB._encoded_size(x.value[]::String, 9)
    end
    return encoded_size
end
