
using LinearAlgebra
using Zygote: jacobian, gradient
using ForwardDiff

function commutation(m, n)
	# translation from Matlab code:
	# https://en.wikipedia.org/wiki/Commutation_matrix
	# determine permutation applied by K
	A = reshape(1:m*n, m, n)
	v = reshape(A', 1, :)
	P = I(m * n)
	P = P[vec(v'), :]
	P
end

function sinkhorn(a, b, C, reg; maxIter = 10, zeroTol = 1e-6)
	M = size(a, 1)
	N = size(b, 1)
	u = ones(M)
	v = ones(N)

	err = 1000.0
	iter = 1

	uKv = 0
	vKTu = 0

	K = exp.(-C ./ reg)

	while ((iter <= maxIter) & (err >= zeroTol))

		u = a ./ (K * v)
		v = b ./ (transpose(K) * u)

		uKv = u .* (K * v)
		vKTu = v .* (transpose(K) * u)

		iter = iter + 1
		err = sum((a .- uKv) .^ 2) + sum((b .- vKTu) .^ 2)
		# println(err)
		# println(iter)
	end

	# cost = 0
	# for i in 1:size(a, 2)
	# 	c = sum(diagm(u[:, i]) * K * diagm(v[:, i]) .* M)
	# 	# println("S: ", c)
	# 	cost += lbd[i] * c
	# end
	# cost
	P = diagm(u) * K * diagm(v)
	sum(P .* C) + reg * sum(P .* (log.(P) .- 1))
	# P
	# P
end

function sinkgrad(a, b, C, reg; maxIter = 10, zeroTol = 1e-6)
	M = size(a, 1)
	N = size(b, 1)
	u = ones(M)
	v = ones(N)
	Ju = zeros(M, M)
	Jv = zeros(N, M)

	err = 1000.0
	iter = 1

	uKv = 0
	vKTu = 0

	K = exp.(-C ./ reg)

	while ((iter <= maxIter) & (err >= zeroTol))
		Kv = K * v
		# display(1 ./ Kv)
		# display(diagm(a ./ (Kv .^ 2)))
		# display(K * Jv)
		Ju = diagm(1 ./ Kv) - diagm(a ./ (Kv .^ 2)) * K * Jv
		u = a ./ Kv

		KTu = K' * u
		Jv = -diagm(b ./ (KTu) .^ 2) * K' * Ju
		v = b ./ KTu

		uKv = u .* (K * v)
		vKTu = v .* (K' * u)
		iter += 1
		# println(iter)
		err = sum((a .- uKv) .^ 2) + sum((b .- vKTu) .^ 2)
	end
	P = diagm(u) * K * diagm(v)

	JLP = ones(M * N)' * diagm(vec(C + reg * log.(P))) # confirmed
	JfJg = kron(ones(N), I(M)) * diagm(1 ./ u) * Ju + kron(I(N), ones(M)) * diagm(1 ./ v) * Jv
	JPa = diagm(vec(P)) * JfJg
	JLa = JLP * JPa
	JLa'
end

function sinkhorn_par(A, B, C, reg; maxIter = 10, zeroTol = 1e-6)
	M = size(a, 1)
	N = size(b, 1)
	D = size(A, 2)
	U = ones(size(A))
	V = ones(size(B))

	err = 1000.0
	iter = 1

	UKV = 0
	VKTU = 0

	K = exp.(-C ./ reg)

	while ((iter <= maxIter) & (err >= zeroTol))
		KV = K * V
		U = A ./ KV

		KTU = K' * U
		V = B ./ KTU

		UKV = U .* (K * V)
		VKTU = V .* (K' * U)
		err = sum((A .- UKV) .^ 2) + sum((B .- VKTU) .^ 2)
		iter += 1
	end
	V
end


function sinkgrad_par(A, B, C, reg; maxIter = 10, zeroTol = 1e-6)
	M = size(a, 1)
	N = size(b, 1)
	D = size(A, 2)
	U = ones(size(A))
	V = ones(size(B))
	JU = zeros(M * D, M * D)
	JV = zeros(N * D, M * D)

	err = 1000.0
	iter = 1

	UKV = 0
	VKTU = 0

	K = exp.(-C ./ reg)

	while ((iter <= maxIter) & (err >= zeroTol))
		KV = K * V
		JU = diagm(1 ./ vec(KV)) - diagm(vec(A ./ (KV .^ 2))) * kron(I(D), K) * JV
		U = A ./ KV

		KTU = K' * U
		JV = -diagm(vec(B ./ (KTU .^ 2))) * kron(I(D), K)' * JU
		V = B ./ KTU

		UKV = U .* (K * V)
		VKTU = V .* (K' * U)
		err = sum((A .- UKV) .^ 2) + sum((B .- VKTU) .^ 2)
		iter += 1
	end
	JV
end


function sinkhorn_log(a, b, C, reg; maxIter = 10, zeroTol = 1e-6)
	M = size(a, 1)
	N = size(b, 1)
	f = zeros(M)
	g = zeros(N)

	onesM = ones(M)
	onesN = ones(N)

	err = 1000.0
	iter = 1

	while ((iter <= maxIter) & (err >= zeroTol))
		S = C - f * onesN' - onesM * g'
		c = minimum(S)
		Q = exp.(-(S .- c) / reg)
		f = f + reg * log.(a) .+ c - reg * log.(Q * onesN)

		S = C - f * onesN' - onesM * g'
		c = minimum(S)
		Q = exp.(-(S .- c) / reg)
		g = g + reg * log.(b) .+ c - reg * log.(Q' * onesM)

		iter += 1
		P = exp.(-(C - f * onesN' - onesM * g') / reg)
		err = sum((P * onesN) .^ 2) + sum((P' * onesM) .^ 2)
	end
	P = exp.(-(C - f * onesN' - onesM * g') / reg)
	sum(P .* C) + reg * sum(P .* (log.(P) .- 1))
end

function sinkgrad_log(a, b, C, reg; maxIter = 10, zeroTol = 1e-6)
	M = Int(size(a, 1))
	N = Int(size(b, 1))
	f = zeros(M)
	g = zeros(N)
	Jf = zeros(M, M)
	Jg = zeros(N, M)

	K = commutation(M, N)
	P = zeros(size(C))

	onesM = ones(M)
	onesN = ones(N)

	err = 1000.0
	iter = 1

	while ((iter <= maxIter) & (err >= zeroTol))
		S = C - f * onesN' - onesM * g'
		c = minimum(S)
		Q = exp.(-(S .- c) / reg)
		J = kron(onesN, I(M)) * Jf + kron(I(N), onesM) * Jg
		Jf = Jf + reg * diagm(1 ./ a) - diagm(vec(1 ./ (Q * onesN))) * kron(onesN, I(M))' * diagm(vec(Q)) * J
		f = f + reg * log.(a) .+ c - reg * log.(Q * onesN)

		S = C - f * onesN' - onesM * g'
		c = minimum(S)
		Q = exp.(-(S .- c) / reg)
		J = kron(onesN, I(M)) * Jf + kron(I(N), onesM) * Jg
		Jg = Jg - diagm(vec(1 ./ (Q' * onesM))) * kron(onesM, I(N))' * K * diagm(vec(Q)) * J
		g = g + reg * log.(b) .+ c - reg * log.(Q' * onesM)

		iter += 1
		P = exp.(-(C - f * onesN' - onesM * g') / reg)
		err = sum((P * onesN) .^ 2) + sum((P' * onesM) .^ 2)
	end
	JPa = diagm(vec(P)) * (kron(onesN, I(M)) * Jf + kron(I(N), onesM) * Jg) / reg
	JLP = ones(M * N)' * diagm(vec(C + reg * log.(P)))
	grad = (JLP * JPa)'
	grad
end

a = [0.6, 0.2, 0.1]
b = [0.3, 0.7]
C = [1 2; 3 4; 5 6]


sinkhorn(a, b, C, 0.1)
sinkhorn(a, b, C, 0.1)
sinkhorn_log(a, b, C, 0.1)

gradient(x -> sinkhorn(x, b, C, 0.1), a)[1]
sinkgrad(a, b, C, 0.1)

gradient(x -> sinkhorn_log(x, b, C, 0.1), a)[1]
sinkgrad_log(a, b, C, 0.1)

jacobian(x -> sinkhorn_par(x, b, C, 0.1), a)[1]
sinkgrad_par(a, b, C, 0.1)





sinkhorn(a, C, b, 0.1)
jacobian(x -> sinkhorn(x, C, b, 0.1, maxIter = 10, zeroTol = 1e-14), a)[1]
sinkgrad(a, C, b, 0.1, maxIter = 10, zeroTol = 1e-14)


gradient(x -> sinkhorn(x, C, b, 0.1, maxIter = 1), a)[1]
sinkgrad(a, C, b, 0.1, maxIter = 1)




P = sinkhorn(a, C, b, 0.1)


jacobian(x -> sum(x .* C) + sum(x .* (log.(x) .- 1)), P)[1]
kron(ones(2), ones(2))' * diagm(vec(C + 0.1 .* log.(P)))






sum(P .* C) + sum(P .* (log.(P) .- 1))
ones(2)' * (P .* (C + log.(P) .- 1)) * ones(2)

sum(P .* (C + log.(P) .- 1))

sum(P)

ones(2)' * P * ones(2)


jacobian(x -> sum(x .* C), P)[1]
ones(4)' * diagm(vec(C ./ 1.0))

kron(ones(3)', ones(2)')
kron(ones(3), ones(2))'



jacobian(x -> sum(x .* C) + 0.1 * sum(x .* (log.(x) .- 1)), P)[1]
ones(4)' * diagm(vec(C + 0.1 * log.(P)))
begin
	ones(4)' * diagm(vec(log.(P) - 1 ./ P + 0.1 * (C .- 1)))
end


jacobian(x -> x, a)[1]



gradient(x -> maximum(x), [1, 2, 3])[1]
X = [1 2 3; 4 6 6]
gradient(x -> maximum(x), [1, 2, 2])[1]

a = [1, 2, 2]

(a.-maximum(a))[1]

gradient(x -> log(sum(exp.(x))), [1, 2, 2])[1]
gradient(x -> maximum(x) + log(sum(exp.(x .- maximum(x)))), [1, 2, 2])[1]
gradient(x -> sum(x .- maximum(x)), [1, 2, 2])[1]
jacobian(x -> exp.(x .- maximum(x)), [1, 2, 2])[1]
jacobian(maximum, [1, 2, 2])[1]
gradient(x -> (x.-maximum(x))[3], a)[1]

gradient(x -> sum(x .- maximum(x)), a)[1]

ForwardDiff.gradient(x -> maximum(x), a)[1]

gradient(maximum, a)[1]
ForwardDiff.gradient(maximum, a)
ForwardDiff.gradient(x -> log(sum(exp.(x))), a)
ForwardDiff.gradient(x -> maximum(x) + log(sum(exp.(x .- maximum(x)))), a)

gradient(x -> log(sum(exp.(x))), a)[1]
gradient(x -> maximum(x) + log(sum(exp.(x .- maximum(x)))), a)[1]

gradient(x -> x / maximum(x), 1.0)
jacobian(x -> x / maximum(x), [1, 2, 3])[1]
gradient(x -> 1 / maximum(x), [1, 2, 2])[1]

a = [1, 2, 2.0]
(x -> 1 ./ maximum(x))(a)
gradient(x -> 1 / maximum(x), a)[1]
ForwardDiff.gradient(x -> 1 ./ maximum(x), a)

gradient(maximum, a)[1]
ForwardDiff.gradient(maximum, a)


a = [1 2 3; 3 4 5]
jacobian(x -> x', a)[1]


x = [1, 2, 3]
ones(3)' * x
ones(3)' * x * ones(3)

function softmax(x)
	expx = exp.(x)
	expx / sum(expx)
end

function softmax2(X)
	N = size(X, 1)
	S = size(X, 2)
	expX = exp.(X)
	expX ./ kron(ones(N)' * expX, ones(N))
end

softmax(x)
softmax2(x)

lbd = [1, 2, 3]
kron(lbd', ones(4))

A = [1 2 3; 3 4 5]

[A * ones(3) A * ones(3)]
[A A] * kron(I(2), ones(3))
