import os
import sys
import time


def calc_prime(n):
	if n < 2:
		return []
	if n == 2:
		return [2]
	# start from 3
	if sys.version_info.major <= 2:
		s = range(3, n + 1, 2)
	else:
		s = list(range(3, n + 1, 2))
	mroot = n ** 0.5
	half = len(s)
	i = 0
	m = 3
	while m <= mroot:
		if s[i]:
			j = (m * m - 3) // 2  # int div
			s[j] = 0
			while j < half:
				s[j] = 0
				j += m
		i = i + 1
		m = 2 * i + 3
	return [2] + [x for x in s if x]

start_time = int(time.time())

res = calc_prime(100000000)
print("Found {} prime numbers.".format(len(res)))
print("Time Elapsed", time.time() - start_time)
