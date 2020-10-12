### A Pluto.jl notebook ###
# v0.12.3

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

# ╔═╡ 3c6ee342-0335-11eb-03a5-5f62fd006afb
md"""## Variables
-  ``M``              - Number of input examples.
-  ``L``              - Number of layers.
-  ``N[l]``           - Number of neurons in layer, ``l``, for ``l\in (1:L)``.
-  ``N[0]``           - Number of features.
-  ``X``              - Input data in an ``N[0]\times M`` matrix.
-  ``W[l]``           - Weights for layer ``l`` in an ``N[l]\times N[l-1]`` matrix.
-  ``b[l]``           - Biases for layer ``l`` as a vector of length ``N[l]``.
-  ``G[l]``           - Activation function for layer ``l``.
-  ``G'[l]``          - Derivative of the activation function for layer ``l``.
- ``Z`` - the weighted sums at each layer, ``l``, with ``Z[l]`` being an ``N[l]\times M`` matrix.
- ``A`` - the activation values at each layer, ``l``, with ``A[l]`` being an ``N[l]\times M`` matrix.
"""


# ╔═╡ c458e23e-0192-11eb-0c01-5ddcc41c3ad6
md"## Network Initialisation"

# ╔═╡ 4bcf08a6-00d5-11eb-2d6f-29ae01e60198
function network_init(X, N, G, G′)
	@assert length(N) == length(G) == length(G′)
	M = size(X, 2)
 	L = length(N)
 	N = OffsetArray([size(X, 1), N...], 0:length(N))
 	W = Vector(undef, L)
	b = Vector(undef, L)
	for l ∈ 1:L
		W[l] = rand(N[l], N[l-1]) * 0.001
		b[l] = zeros(N[l], 1)
	end
	M, L, N, W, b, G, G′
end

# ╔═╡ 558358aa-018b-11eb-0948-6534f0788937
md"## Feedforward
- ``Z`` - the weighted sums at each layer, ``l``, with ``Z[l]`` being an ``N[l]\times M`` matrix.
- ``A`` - the activation values at each layer, ``l``, with ``A[l]`` being an ``N[l]\times M`` matrix.
"

# ╔═╡ 64b7d44a-018b-11eb-31c1-2b77997b0bbe
function feedforward(X, W, b, G)
	L = length(W)
	Z = Vector(undef, L)  # weighted sums
	A = OffsetArray(Vector(undef, L+1), 0:L) # activation values
	A[0] = X
	for l in 1:L		
		Z[l] = W[l]*A[l-1] .+ b[l]
		A[l] = G[l].(Z[l])
	end
	A, Z
end

# ╔═╡ b4077246-0319-11eb-1e83-8bb2598c5d52
md"## Backpropagation"

# ╔═╡ 06b0445a-031a-11eb-253a-693f91000ca1
"```backprop(```
- ```W - weights```
- ```b - biases```
- ```Z = sums```
- ```A - activations```
- ```G' - derivatives of activation functions```
- ```£′ - derivative of cost function```
```)```"
function backprop(W, b, Z, A, G′, £′, Y)
	M = length(A[0][2])
	L = length(W)
	ΔA = similar(A)
	ΔZ = similar(Z)
	ΔW = similar(W)
	Δb = similar(b)
	ΔA[L] = £′(A[end], Y)
	
	@assert M != 0
	@assert !isnan(W[1][1,1])
	for l ∈ L:-1:1

		@assert !isnan(G′[l].(Z[l])[1])

			
		ΔZ[l] = ΔA[l] .* G′[l].(Z[l])
		
		
		ΔW[l] =  ΔZ[l] * A[l-1]' / M
		Δb[l] = sum(ΔZ[l], dims=2) / M
		
		ΔA[l-1] = W[l]' * ΔZ[l]		
	end
	
	ΔW, Δb
end

# ╔═╡ e26ad9f8-0665-11eb-260c-a575fb3cc53d
function update(W, b, ΔW, Δb, λ)
	L = length(W)
	for l ∈ 1:L
		W[l] = W[l] - λ * ΔW[l]
		b[l] = b[l] - λ * Δb[l]
	end
	W, b
end

# ╔═╡ e40b489c-0647-11eb-0493-6faedb88b273
function train(X, Y, W, b, G, G′, £′, λ)
		A, Z = feedforward(X, W, b, G)
		if isnan(A[1][1])
			throw("feedforward")
		end
		ΔW, Δb = backprop(W, b, Z, A, G′, £′, Y)
		if isnan(ΔW[1][1,1])
			throw("backprop")
		end
		W, b = update(W, b, ΔW, Δb, λ)
		W, b
end

# ╔═╡ 761a9cfc-0192-11eb-2798-1743ab4cb424
md"## Activation Functions"

# ╔═╡ cc224ea4-00d6-11eb-0b04-e940cc6374cf
"Sigmoid function."
σ(x) = 1 / (1 + ℯ^-x)

# ╔═╡ f65128ce-031c-11eb-293b-43816c870eaf
"Derivative of the sigmoid function"
σ′(x) = σ(x)*(1 - σ(x))

# ╔═╡ 256deafa-0324-11eb-29b5-2dc1842225a8
md"## Errors
- ``A[L]`` - the predictions for each example as a vector of length N[L].
- ``Y`` - data labels, for each example in a vector of length ``M``, as a onehot vector of length N[L].
- ``£(A, Y)`` - Loss function.
"


# ╔═╡ 513ac7e6-0321-11eb-0545-97e328128433
md"#### Error Functions"

# ╔═╡ faca7322-08b4-11eb-3553-477728c44720
error(Ŷ, Y) = sum((Ŷ - Y).^2) / length(Y)

# ╔═╡ dccf185a-059e-11eb-2142-2fd05d012329
loss′(Ŷ, Y) = Ŷ - Y 

# ╔═╡ 51ad21fa-0638-11eb-1aa6-75fa8be40f78


# ╔═╡ 474b29f0-000c-11eb-1f92-d9dd6686684a
md"## Data Preparation"

# ╔═╡ 20eba704-000e-11eb-11eb-c7ef007de85d
MNIST.convert2image(MNIST.traintensor(1))

# ╔═╡ 57e94614-059f-11eb-0796-5f580b21e9dd
md"## Application"

# ╔═╡ 24a79d26-018c-11eb-219c-f1e2fa8de350
#A, Z = feedforward(train_x, W, b, G);

# ╔═╡ b29597fe-01a1-11eb-193d-27718af41693
# Quick Check
# let
# 	for l in 1:L
# 		@assert size(A[l]) == (N[l], M)
# 	end
# end

# ╔═╡ 21ff3b1c-0018-11eb-3a73-57b0a960523b
"""
```flatten_images(image_data)```

Reshapes each image in ```image_data``` to be 1-dimensional.

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

# ╔═╡ 86f50c3e-033a-11eb-16b1-ed4974ac8c07
"""
```
onehotify(labels)
```
Convert mnist labels into onehot form. (num_features x M)

"""
function onehotify(labels)
	labels .+= 1
	M = length(labels)
	hots = zeros(Int, 10, M)
	for m in 1:M
		hots[labels[m], m] = 1
	end
	hots
end

# ╔═╡ bca69342-000c-11eb-1ab6-1daa2ee00a65
# This currently this fails if the MNIST data hasn't been downloaded. The problem can be seen by looking at output in the terminal in which Pluto was run.
# The solution is to run the following in a Julia repl:
# 	using MLDatasets
# 	MNIST.download()
begin
	train_x, train_y = MNIST.traindata()	
	test_x,  test_y  = MNIST.testdata()
	
	train_x = flatten_images(train_x)
	train_y = onehotify(train_y)
	
	test_x = flatten_images(test_x)
	test_y = onehotify(test_y)
	
end;

# ╔═╡ 916adc44-08a9-11eb-25ae-15b847cb4b74
MNIST.convert2image(train_x[:,1])


# ╔═╡ 0607fd20-08c6-11eb-1011-4d4675bfb207
size(train_x[1])

# ╔═╡ 55515a98-00d3-11eb-3340-031b9ea97166
let
	iterations = 1000
	range = 1:100
	X = train_x[:,range]
	Y = train_y[:,range]
	λ = 0.01
	M, L, N, W, b, G, G′ = network_init(X, [300, 10], [σ, σ], [σ′, σ′])
	£′ = loss′
	Err = error
	errors = []
	weights = []
	#A, Z = feedforward(X, W, b, G)
	for i ∈ 1:iterations
		W, b = train(X, Y, W, b, G, G′, £′, λ)
		
		if i % 100 ==  0
			A, Z = feedforward(X, W, b, G)
			push!(errors, Err(A[end], Y)) 
		end
	end
	errors
end

# ╔═╡ 7c683864-01a4-11eb-279d-450ec26af36e
todo(text) = Markdown.MD(Markdown.Admonition("tip", "TODO", [text]))

# ╔═╡ 5b3d19b8-0661-11eb-1f97-cfdcea7efe8b
todo(md"Stop using \prime")

# ╔═╡ 4c76b1e6-031e-11eb-2c79-65bc171fe56f
todo(md"Derive the ``σ(x)(1-σ(x))`` form for ``σ'(x)``.")

# ╔═╡ Cell order:
# ╟─18e6adaa-000c-11eb-1244-bbd8a68d669d
# ╟─e4aee010-0013-11eb-2f4c-a56467ae908d
# ╠═cbe8a3f6-0014-11eb-198b-f32ad4453e5e
# ╠═ef8fa058-0013-11eb-37f8-31244f0d2ffa
# ╠═3c6ee342-0335-11eb-03a5-5f62fd006afb
# ╟─c458e23e-0192-11eb-0c01-5ddcc41c3ad6
# ╠═4bcf08a6-00d5-11eb-2d6f-29ae01e60198
# ╟─558358aa-018b-11eb-0948-6534f0788937
# ╠═64b7d44a-018b-11eb-31c1-2b77997b0bbe
# ╟─b4077246-0319-11eb-1e83-8bb2598c5d52
# ╠═06b0445a-031a-11eb-253a-693f91000ca1
# ╠═e26ad9f8-0665-11eb-260c-a575fb3cc53d
# ╟─5b3d19b8-0661-11eb-1f97-cfdcea7efe8b
# ╠═e40b489c-0647-11eb-0493-6faedb88b273
# ╟─761a9cfc-0192-11eb-2798-1743ab4cb424
# ╠═cc224ea4-00d6-11eb-0b04-e940cc6374cf
# ╠═f65128ce-031c-11eb-293b-43816c870eaf
# ╟─4c76b1e6-031e-11eb-2c79-65bc171fe56f
# ╟─256deafa-0324-11eb-29b5-2dc1842225a8
# ╟─513ac7e6-0321-11eb-0545-97e328128433
# ╠═faca7322-08b4-11eb-3553-477728c44720
# ╠═dccf185a-059e-11eb-2142-2fd05d012329
# ╟─51ad21fa-0638-11eb-1aa6-75fa8be40f78
# ╟─474b29f0-000c-11eb-1f92-d9dd6686684a
# ╠═20eba704-000e-11eb-11eb-c7ef007de85d
# ╠═bca69342-000c-11eb-1ab6-1daa2ee00a65
# ╠═916adc44-08a9-11eb-25ae-15b847cb4b74
# ╠═0607fd20-08c6-11eb-1011-4d4675bfb207
# ╟─57e94614-059f-11eb-0796-5f580b21e9dd
# ╠═24a79d26-018c-11eb-219c-f1e2fa8de350
# ╠═b29597fe-01a1-11eb-193d-27718af41693
# ╠═55515a98-00d3-11eb-3340-031b9ea97166
# ╟─21ff3b1c-0018-11eb-3a73-57b0a960523b
# ╟─86f50c3e-033a-11eb-16b1-ed4974ac8c07
# ╟─7c683864-01a4-11eb-279d-450ec26af36e
