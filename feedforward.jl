### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ ef8fa058-0013-11eb-37f8-31244f0d2ffa
begin
	using MLDatasets
	using ImageCore
	using OffsetArrays
	using Images
end

# ╔═╡ 18e6adaa-000c-11eb-1244-bbd8a68d669d
md"# A Feedforward Network In Julia
Tested on the mnist digit data"

# ╔═╡ e4aee010-0013-11eb-2f4c-a56467ae908d
md"## Imports"

# ╔═╡ cbe8a3f6-0014-11eb-198b-f32ad4453e5e
begin
	import Pkg
	Pkg.add(["MLDatasets", "ImageCore", "OffsetArrays"])
	Pkg.add(["Images", "ImageIO", "ImageMagick"])
end

# ╔═╡ 474b29f0-000c-11eb-1f92-d9dd6686684a
md"## Data"

# ╔═╡ bca69342-000c-11eb-1ab6-1daa2ee00a65
# This currently this fails if the MNIST data hasn't been downloaded. The problem can be seen by looking at output in the terminal in which Pluto was run.
# The solution is to run the following in a Julia repl:
# 	using MLDatasets
# 	MNIST.download()
begin
	train_x, train_y = MNIST.traindata()	
	test_x,  test_y  = MNIST.testdata()
	ElementType = typeof(train_x[1,1,1])
end;

# ╔═╡ 20eba704-000e-11eb-11eb-c7ef007de85d
MNIST.convert2image(MNIST.traintensor(1))

# ╔═╡ 21ff3b1c-0018-11eb-3a73-57b0a960523b
"""
Reshapes each image in image_data to be 1-dimensional.

```
size(image_data) == rows, cols, M 

size(flatten(image_data)) == rows*cols, M
```
where ```rows``` and ```cols``` are the image dimensions and ```M``` is the number of examples.
"""
function flatten_images(image_data)
	rows, cols, M = size(image_data)
	reshape(image_data, rows*cols, M)
end

# ╔═╡ 6adf3d48-002c-11eb-176e-7d6a52ceaffa
md"## Variables"

# ╔═╡ a210aa7a-00ca-11eb-3ff1-279e9ef75aea
md"
- ``M`` - number of input examples
- ``L`` - number of layers
- ``N[l]`` - number of neurons in layer ``l``, for ``l\in(1:L)``
- ``N[0]`` - number of features
- ``X`` - input data, in an ``N[0]\times M`` matrix
- ``W`` - weight matrices, in a vector of length ``L``, with ``size(W[l])=(N[l], N[l-1])``   
- ``b`` - biases, in a vector of length ``L``.
- ``F`` - activation functions, one for each layer (``F[l]`` refers to the activation function for layer ``l``).
"

# ╔═╡ 761a9cfc-0192-11eb-2798-1743ab4cb424
md"## Activation Functions"

# ╔═╡ cc224ea4-00d6-11eb-0b04-e940cc6374cf
"Sigmoid function."
σ(x)::ElementType = 1 / (1 + ℯ^-x)

# ╔═╡ c458e23e-0192-11eb-0c01-5ddcc41c3ad6
md"## Network Initialisation"

# ╔═╡ 4bcf08a6-00d5-11eb-2d6f-29ae01e60198
function network_init(data, layer_sizes, activation_functions)
	X = data
	M = size(data, 2)
 	L = length(layer_sizes)
 	N = OffsetArray([size(data, 1), layer_sizes...], 0:length(layer_sizes))
 	W = Vector(undef, L)
	b = Vector(undef, L)
	for l ∈ 1:L
		W[l] = rand(ElementType, N[l], N[l-1])
		b[l] = rand()
	end
	F = activation_functions
	X, M, L, N, W, b, F
end

# ╔═╡ 55515a98-00d3-11eb-3340-031b9ea97166
begin
	data = flatten_images(train_x)
	layer_sizes = [30, 50, 10]
	activation_functions = [σ, σ, σ]
	X, M, L, N, W, b = network = network_init(data, layer_sizes, activation_functions)
end;

# ╔═╡ 558358aa-018b-11eb-0948-6534f0788937
md"## Feedforward"

# ╔═╡ ca8bbc52-01a7-11eb-2803-516ab6a98ade
md"
- ``Z`` - the weighted sums at each layer
- ``A`` - the activation values at each layer, ``size(A[l]) = (N[l], M)``
"

# ╔═╡ 64b7d44a-018b-11eb-31c1-2b77997b0bbe
function feedforward(X, M, L, N, W, b, F)
	Z = Vector(undef, L)  # weighted sums
	A = OffsetArray(Vector(undef, L+1), 0:L) # activation values
	A[0] = X
	for l in 1:L		
		Z[l] = W[l]*A[l-1]
		A[l] = F[l].(Z[l] .+ b[l])
	end
	Z, A
end

# ╔═╡ 24a79d26-018c-11eb-219c-f1e2fa8de350
Z, A = feedforward(network...);

# ╔═╡ c0e4f0a0-01a3-11eb-3773-8dd03f13a974
md"#### Sanity Checks"

# ╔═╡ b29597fe-01a1-11eb-193d-27718af41693
begin
	for l in 1:L
		@assert size(A[l]) == (N[l], M)
	end
	@assert typeof(A[1][1,1]) == ElementType
	@assert typeof(Z[1][1]) == ElementType
end

# ╔═╡ 7c683864-01a4-11eb-279d-450ec26af36e
todo(text) = Markdown.MD(Markdown.Admonition("tip", "TODO", [text]))

# ╔═╡ 5512e07a-01a4-11eb-2fbc-258c7ebd3973
todo("Try different types for ElementType!")

# ╔═╡ Cell order:
# ╟─18e6adaa-000c-11eb-1244-bbd8a68d669d
# ╟─e4aee010-0013-11eb-2f4c-a56467ae908d
# ╠═cbe8a3f6-0014-11eb-198b-f32ad4453e5e
# ╠═ef8fa058-0013-11eb-37f8-31244f0d2ffa
# ╟─474b29f0-000c-11eb-1f92-d9dd6686684a
# ╠═bca69342-000c-11eb-1ab6-1daa2ee00a65
# ╟─5512e07a-01a4-11eb-2fbc-258c7ebd3973
# ╠═20eba704-000e-11eb-11eb-c7ef007de85d
# ╠═21ff3b1c-0018-11eb-3a73-57b0a960523b
# ╟─6adf3d48-002c-11eb-176e-7d6a52ceaffa
# ╟─a210aa7a-00ca-11eb-3ff1-279e9ef75aea
# ╟─761a9cfc-0192-11eb-2798-1743ab4cb424
# ╠═cc224ea4-00d6-11eb-0b04-e940cc6374cf
# ╟─c458e23e-0192-11eb-0c01-5ddcc41c3ad6
# ╠═4bcf08a6-00d5-11eb-2d6f-29ae01e60198
# ╠═55515a98-00d3-11eb-3340-031b9ea97166
# ╟─558358aa-018b-11eb-0948-6534f0788937
# ╠═ca8bbc52-01a7-11eb-2803-516ab6a98ade
# ╠═64b7d44a-018b-11eb-31c1-2b77997b0bbe
# ╠═24a79d26-018c-11eb-219c-f1e2fa8de350
# ╟─c0e4f0a0-01a3-11eb-3773-8dd03f13a974
# ╠═b29597fe-01a1-11eb-193d-27718af41693
# ╟─7c683864-01a4-11eb-279d-450ec26af36e
