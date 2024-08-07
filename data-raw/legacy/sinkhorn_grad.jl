# using Zygote AD library to check the Sinkhorn gradient

using Zygote: gradient, jacobian
using LinearAlgebra

a = [0.4  0.3  0.2; 0.5  0.2  0.2; 0.1  0.5  0.6]
b = [0.3; 0.3; 0.4]
M = [0.1  0.3  0.2; 0.5  0.1  0.5; 0.4  0.2  0.1]
lbd = [0.2; 0.2; 0.6]

function sinkhorn(a, lbd, M, b, reg)
	u = ones(size(b, 1), size(a, 2))
	v = ones(size(b, 1), size(a, 2))

	err = 1000.0
	iter = 0

	uKv = 0
	vKTu = 0

	K = exp.(-M ./ reg)

	while ((iter < 1000) & (err >= 1e-10))

		u = a ./ (K * v)
		v = b ./ (transpose(K) * u)

		uKv = u .* (K * v)
		vKTu = v .* (transpose(K) * u)

		iter = iter + 1
		err = sum((a .- uKv) .^ 2) + sum((b .- vKTu) .^ 2)
		# println(err)
	end

	cost = 0
	for i in 1:size(a, 2)
		c = sum(diagm(u[:, i]) * K * diagm(v[:, i]) .* M)
		# println("S: ", c)
		cost += lbd[i] * c
	end
	cost
end

function sinkhorn_g(a, lbd, M, b, reg)
	u = ones(size(b, 1), size(a, 2))
	v = ones(size(b, 1), size(a, 2))

	err = 1000.0
	iter = 0

	uKv = 0
	vKTu = 0

	K = exp.(-M ./ reg)

	while ((iter < 1000) & (err >= 1e-10))

		u = a ./ (K * v)
		v = b ./ (transpose(K) * u)

		uKv = u .* (K * v)
		vKTu = v .* (transpose(K) * u)

		iter = iter + 1
		err = sum((a .- uKv) .^ 2) + sum((b .- vKTu) .^ 2)
		# println(err)
	end

	cost = 0
	for i in 1:size(a, 2)
		c = sum(diagm(u[:, i]) * K * diagm(v[:, i]) .* M)
		# println("S: ", c)
		# cost += lbd[i] * c
		cost += c
	end
	cost
end

sinkhorn(a, lbd, M, b, 0.1)

# sinkhorn gradient wrt lbd
gradient(sinkhorn, a, lbd, M, b, 0.1)[2]
# sinkhorn gradient wrt a
gradient(sinkhorn, a, lbd, M, b, 0.1)[1]
