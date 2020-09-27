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
md"### Definitions
- ``M`` - number of input examples
- ``L`` - number of layers
- ``N[l]`` - number of neurons in layer ``l``, for ``l\in(1:L)``
- ``N[0]`` - number of features
- ``X`` - input data in an ``N[0]\times M`` matrix
- ``W`` - weight matrices of sizes ``N[l]\times N[l-1]`` in an array of length ``L`` 
"

# ╔═╡ 4bcf08a6-00d5-11eb-2d6f-29ae01e60198
function network_init(data, layer_sizes)
	M = size(data, 2)
 	L = length(layer_sizes)
 	N = OffsetArray([size(data, 1), layer_sizes...], 0:length(layer_sizes))
 	X = data
 	W = []
	b = []
	for l ∈ 1:L
		push!(W, rand(N[l], N[l-1]))
		push!(b, rand())
	end
	M, L, N, X, W, b
end

# ╔═╡ cc224ea4-00d6-11eb-0b04-e940cc6374cf
data = flatten_images(train_x);

# ╔═╡ 55515a98-00d3-11eb-3340-031b9ea97166
M, L, N, X, W, b = network_init(data, [30, 50, 10]);

# ╔═╡ Cell order:
# ╟─18e6adaa-000c-11eb-1244-bbd8a68d669d
# ╟─e4aee010-0013-11eb-2f4c-a56467ae908d
# ╠═cbe8a3f6-0014-11eb-198b-f32ad4453e5e
# ╠═ef8fa058-0013-11eb-37f8-31244f0d2ffa
# ╟─474b29f0-000c-11eb-1f92-d9dd6686684a
# ╠═bca69342-000c-11eb-1ab6-1daa2ee00a65
# ╠═20eba704-000e-11eb-11eb-c7ef007de85d
# ╠═21ff3b1c-0018-11eb-3a73-57b0a960523b
# ╟─6adf3d48-002c-11eb-176e-7d6a52ceaffa
# ╟─a210aa7a-00ca-11eb-3ff1-279e9ef75aea
# ╠═4bcf08a6-00d5-11eb-2d6f-29ae01e60198
# ╠═cc224ea4-00d6-11eb-0b04-e940cc6374cf
# ╠═55515a98-00d3-11eb-3340-031b9ea97166
