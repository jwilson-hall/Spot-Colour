#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <time.h>
using namespace std;

void calc_prime(int n, vector<int> &res) {
	if (n < 2) return;
	if (n == 2) {
		res.push_back(2);
		return;
	}
	vector<int> s;
	
	for (int i = 3; i < n + 1; i += 2) {
		s.push_back(i);
		
	}
	int mroot = sqrt(n);
	int half = static_cast<int>(s.size());
	int i = 0;
	int m = 3;
	while (m <= mroot) {
		if (s[i]) {
			int j = static_cast<int>((m*m - 3)*0.5);
			s[j] = 0;
			while (j < half) {
				s[j] = 0;
				j += m;
			}
		}
		i = i + 1;
		m = 2*i + 3;
	}
	res.push_back(2);	

	std::vector<int>::iterator pend = std::remove(s.begin(), s.end(), 0);
	res.insert(res.begin() + 1, s.begin(), pend);
}

int main() {
	std::time_t startTime = std::time(NULL);	
	clock_t begin = clock();

	vector<int> res;
	calc_prime(100000000, res);
	clock_t end = clock();
	std::time_t endTime = std::time(NULL);
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("Found %d prime numbers.\n", (int)res.size());
	printf("Time Elapsed: %.20f",time_spent);

	return 0;
}
