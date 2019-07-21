module VectorFieldsOnASphere

using LabelledArrays,OffsetArrays

import LinearAlgebra: norm,normalize,dot
import Base: +,-,*,/,==,≈,collect

using PointsOnASphere
import TwoPointFunctions: cosχ

export CartesianVector,SphericalVector,HelicityVector,SphericalPolarVector,unitvector

abstract type VectorField end

const vector_types = (:CartesianVector,:SphericalVector,:SphericalPolarVector,:HelicityVector)

function cartesian_components(x,y,z) 
	v = @LVector Float64 (:x,:y,:z)
	v.x = x
	v.y = y
	v.z = z
	return v
end

function spherical_components(p,z,m)
	v = @LVector ComplexF64 (:p,:z,:m)
	v.p = p
	v.z = z
	v.m = m
	return v
end

function spherical_polar_components(r,θ,ϕ)
	v = @LVector Float64 (:r,:θ,:ϕ)
	v.r = r
	v.θ = θ
	v.ϕ = ϕ
	return v
end

helicity_components(p,z,m) = spherical_components(p,z,m)

# Vector fields evaluated at a point
struct CartesianVector <: VectorField
	components :: AbstractArray
	pt::Point3D
end

struct SphericalVector <: VectorField
	components :: AbstractArray
	pt::Point3D
end

struct SphericalPolarVector <: VectorField
	components :: AbstractArray
	pt::Point3D
end

struct HelicityVector <: VectorField
	components :: AbstractArray
	pt::Point3D
end

const complex_basis = Union{SphericalVector,HelicityVector}
const real_basis = Union{CartesianVector,SphericalPolarVector}
const fixed_basis = Union{CartesianVector,SphericalVector}
const rotating_basis = Union{SphericalPolarVector,HelicityVector}

const CtoS = (@SLArray ComplexF64 (3,3) (:px,:zx,:mx,:py,:zy,:my,:pz,:zz,:mz))(
								-1/√2,	0,	1/√2,
								im/√2,	0,	im/√2,
									0,	1,	0)

const StoC = (@SLArray ComplexF64 (3,3) (:xp,:yp,:zp,:xz,:yz,:zz,:xm,:ym,:zm))(CtoS')
const SPtoH = (@SLArray ComplexF64 (3,3) (:pθ,:zθ,:mθ,:pϕ,:zϕ,:mϕ,:pr,:zr,:mr))(CtoS)
const HtoSP = (@SLArray ComplexF64 (3,3) (:θp,:ϕp,:rp,:θz,:ϕz,:rz,:θm,:ϕm,:rm))(StoC)

function SP_to_C_matrix(v)
	(@SLArray Float64 (3,3) (:xr,:yr,:zr,:xθ,:yθ,:zθ,:xϕ,:yϕ,:zϕ))(
		[sin(v.pt.θ)cos(v.pt.ϕ)	cos(v.pt.θ)cos(v.pt.ϕ)	-sin(v.pt.ϕ)
		 sin(v.pt.θ)sin(v.pt.ϕ)	cos(v.pt.θ)sin(v.pt.ϕ)	cos(v.pt.ϕ)
		 cos(v.pt.θ)			-sin(v.pt.θ)					0])
end

function C_to_SP_matrix(v)
	(@SLArray Float64 (3,3) (:rx,:θx,:ϕx,:ry,:θy,:ϕy,:rz,:θz,:ϕz))(SP_to_C_matrix(v)')
end

# Matrix to transform from cartesian to spherical coordinates
function CartesianToSpherical(v::CartesianVector)
	components = @LVector ComplexF64 (:p, :m, :z)
	vc = v.components
	components.z = vc.z
	components.p = CtoS.px * vc.x + CtoS.py * vc.y
	components.m = CtoS.mx * vc.x + CtoS.my * vc.y
	SphericalVector(components,v.pt)
end

function CartesianToSphericalPolar(v::CartesianVector)
	M = C_to_SP_matrix(v)
	components = @LVector Float64 (:r, :θ, :ϕ)
	vc = v.components
	components.r = M.rx*vc.x + M.ry*vc.y + M.rz*vc.z
	components.θ = M.θx*vc.x + M.θy*vc.y + M.θz*vc.z
	components.ϕ = M.ϕx*vc.x + M.ϕy*vc.y + M.ϕz*vc.z

	SphericalPolarVector(components,v.pt)
end

CartesianToHelicity(v::CartesianVector) = SphericalPolarToHelicity(CartesianToSphericalPolar(v))

function SphericalToCartesian(v::SphericalVector)
	components = @LVector Float64 (:x, :y, :z)
	vc = v.components
	components.x = real(StoC.xp*vc.p + StoC.xm*vc.m)
	components.y = real(StoC.yp*vc.p + StoC.ym*vc.m)
	components.z = real(vc.z)
	CartesianVector(components,v.pt)
end

function SphericalToSphericalPolar(v::SphericalVector)
	CartesianToSphericalPolar(SphericalToCartesian(v))
end

SphericalToHelicity(v::SphericalVector) = SphericalPolarToHelicity(SphericalToSphericalPolar(v))

function SphericalPolarToCartesian(v::SphericalPolarVector)
	M = SP_to_C_matrix(v)
	components = cartesian_components(zero(Float64),zero(Float64),zero(Float64))
	vc = v.components
	components.x = M.xr*vc.r + M.xθ*vc.θ + M.xϕ*vc.ϕ
	components.y = M.yr*vc.r + M.yθ*vc.θ + M.yϕ*vc.ϕ
	components.z = M.zr*vc.r + M.zθ*vc.θ + M.zϕ*vc.ϕ
	CartesianVector(components,v.pt)
end

function SphericalPolarToSpherical(v::SphericalPolarVector)
	CartesianToSpherical(SphericalPolarToCartesian(v))
end

function SphericalPolarToHelicity(v::SphericalPolarVector)
	components = @LVector ComplexF64 (:p, :m, :z)
	vc = v.components
	components.p = SPtoH.pθ*vc.θ + SPtoH.pϕ*vc.ϕ
	components.m = SPtoH.mθ*vc.θ + SPtoH.mϕ*vc.ϕ
	components.z = vc.r
	HelicityVector(components,v.pt)
end

HelicityToCartesian(v::HelicityVector) = SphericalPolarToCartesian(HelicityToSphericalPolar(v))

HelicityToSpherical(v::HelicityVector) = SphericalPolarToSpherical(HelicityToSphericalPolar(v))

function HelicityToSphericalPolar(v::HelicityVector)
	components = @LVector Float64 (:r, :θ, :ϕ)
	vc = v.components
	components.θ = real(HtoSP.θp*vc.p + HtoSP.θm*vc.m)
	components.ϕ = real(HtoSP.ϕp*vc.p + HtoSP.ϕm*vc.m)
	components.r = real(vc.z)
	SphericalPolarVector(components,v.pt)
end


# Assume radius vector by default if coordinates are not specified
SphericalPolarVector(x::Point3D) = SphericalPolarVector(spherical_polar_components(x.r,0,0),x)
# Assume point on unit sphere if distance from origin is not specified
SphericalPolarVector(n::Point2D) = SphericalPolarVector(spherical_polar_components(1,0,0),Point3D(1,n))

CartesianVector(x::SphericalPoint) = SphericalPolarToCartesian(SphericalPolarVector(x))
SphericalVector(x::SphericalPoint) = SphericalPolarToSpherical(SphericalPolarVector(x))
HelicityVector(x::SphericalPoint) = SphericalPolarToHelicity(SphericalPolarVector(x))

CartesianVector(v::CartesianVector) = v
CartesianVector(v::SphericalVector) = SphericalToCartesian(v)
CartesianVector(v::SphericalPolarVector) = SphericalPolarToCartesian(v)
CartesianVector(v::HelicityVector) = HelicityToCartesian(v)

HelicityVector(v::HelicityVector) = v
HelicityVector(v::CartesianVector) = CartesianToHelicity(v)
HelicityVector(v::SphericalVector) = SphericalToHelicity(v)
HelicityVector(v::SphericalPolarVector) = SphericalPolarToHelicity(v)

SphericalVector(v::SphericalVector) = v
SphericalVector(v::CartesianVector) = CartesianToSpherical(v)
SphericalVector(v::SphericalPolarVector) = SphericalPolarToSpherical(v)
SphericalVector(v::HelicityVector) = HelicityToSpherical(v)

SphericalPolarVector(v::SphericalPolarVector) = v
SphericalPolarVector(v::CartesianVector) = CartesianToSphericalPolar(v)
SphericalPolarVector(v::SphericalVector) = SphericalToSphericalPolar(v)
SphericalPolarVector(v::HelicityVector) = HelicityToSphericalPolar(v)

twoD_types = Union{Point2D,Tuple{<:Real,<:Real}}
threeD_types = Union{Point3D,Tuple{<:Real,<:Real,<:Real}}

CartesianVector(x,y,z,pt::threeD_types) = CartesianVector(cartesian_components(x,y,z),Point3D(pt))
SphericalVector(p,z,m,pt::threeD_types) = SphericalVector(spherical_components(p,z,m),Point3D(pt))
SphericalPolarVector(r,θ,ϕ,pt::threeD_types) = SphericalPolarVector(spherical_polar_components(r,θ,ϕ),Point3D(pt))
HelicityVector(p,z,m,pt::threeD_types) = HelicityVector(helicity_components(p,z,m),Point3D(pt))

CartesianVector(x,y,z,n::twoD_types) = CartesianVector(x,y,z,Point3D(1,n))
SphericalVector(p,z,m,n::twoD_types) = SphericalVector(p,z,m,Point3D(1,n))
SphericalPolarVector(r,θ,ϕ,n::twoD_types) = SphericalPolarVector(r,θ,ϕ,Point3D(1,n))
HelicityVector(p,z,m,n::twoD_types) = HelicityVector(p,z,m,Point3D(1,n))

for T in vector_types
	# Iterables
	@eval $(T)(v,n::SphericalPoint) = $(T)(v[1],v[2],v[3],n)
end

(==)(v1::T,v2::T) where T<:VectorField = v1 === v2
(≈)(v1::T,v2::T) where T<:VectorField = (v1.components ≈ v2.components) && (v1.pt === v2.pt)

norm(v::T) where T<:real_basis = norm(v.components)
function norm(v::T) where T<:complex_basis
	real(√( v.components.z^2 - 2v.components.p*v.components.m ))
end

normalize(v::T) where T<:real_basis = v/norm(v)

normalize(v::SphericalVector) = SphericalVector(normalize(CartesianVector(v)))
normalize(v::HelicityVector) = HelicityVector(normalize(SphericalPolarVector(v)))

unitvector(v::VectorField) = normalize(v)

(+)(v1::T,v2::T) where T<:fixed_basis = T(v1.components + v2.components,v1.pt)
(-)(v1::T,v2::T) where T<:fixed_basis = T(v1.components - v2.components,v2.pt)

function (+)(v1::T,v2::T) where T<:rotating_basis
	if v1.pt === v2.pt
		return T(v1.components + v2.components,v1.pt)
	elseif (v1.pt.θ == v2.pt.θ) && (v1.pt.ϕ == v2.pt.ϕ)
		# in this case the points are radially separated, 
		# and the basis vectors are aligned
		return T(v1.components + v2.components,v1.pt)
	else
		v1C = CartesianVector(v1)
		v2C = CartesianVector(v2)
		T(v1C + v2C)
	end
end

function (-)(v1::T,v2::T) where T<:rotating_basis
	if v1.pt === v2.pt
		return T(v1.components - v2.components,v2.pt)
	elseif (v1.pt.θ == v2.pt.θ) && (v1.pt.ϕ == v2.pt.ϕ)
		# in this case the points are radially separated, 
		# and the basis vectors are aligned
		return T(v1.components - v2.components,v2.pt)
	else
		v1C = CartesianVector(v1)
		v2C = CartesianVector(v2)
		T(v1C - v2C)
	end
end

# Scaling
(*)(v::T,c::Number) where T<:VectorField = T(v.components.*c,v.pt)
(*)(c::Number,v::T) where T<:VectorField = T(v.components.*c,v.pt)
(/)(v::T,c::Number) where T<:VectorField = T(v.components./c,v.pt)

cosχ(v::VectorField,w::VectorField) = cosχ(v.pt,w.pt)

# Inner product of two vectors
function dot(v::VectorField,w::VectorField)
	# Use the fact that a⋅b = |a||b|cos(θ)
	norm(v)*norm(w)*cosχ(v,w)
end

Base.parent(v::VectorField) = v.components
Base.getindex(v::VectorField,inds...) = getindex(parent(v),inds...)

# convert to array
collect(v::real_basis) = collect(v.components)
collect(v::complex_basis) = OffsetArray([v.components.m,v.components.z,v.components.p],-1:1)

# display
function Base.show(io::IO, v::VectorField)
    compact = get(io, :compact, false)

    print(io,round.(v.components,sigdigits=3)," at $(v.pt)")
end

function Base.show(io::IO, ::MIME"text/plain", v::VectorField)
    println(io,"Vector with components $(v.components)")
    print(io,"Defined at $(v.pt)")
end


end # module
