function calc_prime(n) {
	if (n < 2) { return []; }
	if (n == 2) { return [2]; }

	var s = [];
	for (var i = 3; i < n + 1; i += 2) {
		s.push(i);
	}

	var mroot = Math.floor(Math.sqrt(n));
	var half = s.length;
	var i = 0;
	var m = 3;

	while (m <= mroot) {
		if (s[i]) {
			var j = Math.floor((m*m-3)/2);   // int div
			s[j] = 0;
			while (j < half) {
				s[j] = 0;
				j += m;
			}
		}
		i = i + 1;
		m = 2*i + 3;
	}

	var res = [];
	res.push(2);

	for (var x = 0; x < s.length; x++) {
		if (s[x]) {
			res.push(s[x]);
		}
	}
	return res;
}

var startTime = Date.now();
var res = calc_prime(100000000);
var endTime = Date.now()-startTime;
console.log("Found " + res.length + " prime numbers.");
console.log("Time Elapsed:  " + endTime );

