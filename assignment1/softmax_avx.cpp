#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>      
#include <hpc_helpers.hpp>
#include <avx_mathfun.h>


void softmax_avx(const float *input, float *output, size_t K) {
	// Find the maximum to stabilize the computation of the exponential

    float max_val = -std::numeric_limits<float>::infinity();
    size_t i = 0;
	// inizialize a vector of 8 float elements at -infty
    __m256 max_vec = _mm256_set1_ps(max_val);

    for (; i + 8 <= K; i += 8) {
		// load 8 float from memory in a vector
        __m256 in_vec = _mm256_loadu_ps(&input[i]);
		// compare each element to find the maximum value o
        max_vec = _mm256_max_ps(max_vec, in_vec);
    }

    float temp[8];
    _mm256_storeu_ps(temp, max_vec); // save the AVX vactor in an array to do the comparison to find the maximum of all elements
    for (int j = 0; j < 8; j++) {
        max_val = std::max(max_val, temp[j]); 
    }

	// check others elements if k is not multiple of 8
    for (; i < K; ++i) {
        max_val = std::max(max_val, input[i]);
    }

    // Computes all exponentials with the shift of max_val and the total sum
	

    i = 0;
    __m256 sum_vec = _mm256_setzero_ps(); // all zero vector of 8 float for the esponential sum

    for (; i + 8 <= K; i += 8) {
        __m256 in_vec = _mm256_loadu_ps(&input[i]);
        __m256 shifted = _mm256_sub_ps(in_vec, _mm256_set1_ps(max_val)); // subtracts the maximum for numeric stability
        __m256 exp_vec = exp256_ps(shifted); // compute the exponential
        _mm256_storeu_ps(&output[i], exp_vec); // saves the exponentials
        sum_vec = _mm256_add_ps(sum_vec, exp_vec); // accumulates the sum
    }

	// extraction of the sum
    float temp_sum[8];
    _mm256_storeu_ps(temp_sum, sum_vec);
    float sum = 0.0f;
    for (int j = 0; j < 8; j++) {
        sum += temp_sum[j]; // sum all values
    }

	// check for the others elements
    for (; i < K; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }

	// calculation of the inverse of the sum to normalize
    i = 0;
    __m256 sum_vec_inv = _mm256_set1_ps(1.0f / sum);

    for (; i + 8 <= K; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&output[i]); // load the exponential values
        out_vec = _mm256_mul_ps(out_vec, sum_vec_inv); // divide for the sum
        _mm256_storeu_ps(&output[i], out_vec); // save the result
    }

	// check for the other elements
    for (; i < K; ++i) {
        output[i] /= sum;
    }
}


std::vector<float> generate_random_input(size_t K, float min = -1.0f, float max = 1.0f) {
    std::vector<float> input(K);
    //std::random_device rd;
    //std::mt19937 gen(rd());
	std::mt19937 gen(5489); // fixed seed for reproducible results
    std::uniform_real_distribution<float> dis(min, max);
    for (size_t i = 0; i < K; ++i) {
        input[i] = dis(gen);
    }
    return input;
}

void printResult(std::vector<float> &v, size_t K) {
	for(size_t i=0; i<K; ++i) {
		std::fprintf(stderr, "%f\n",v[i]);
	}
}


int main(int argc, char *argv[]) {
	if (argc == 1) {
		std::printf("use: %s K [1]\n", argv[0]);
		return 0;		
	}
	size_t K=0;
	if (argc >= 2) {
		K = std::stol(argv[1]);
	}
	bool print=false;
	if (argc == 3) {
		print=true;
	}	
	std::vector<float> input=generate_random_input(K);
	std::vector<float> output(K);

	std::cout << K;

	TIMERSTART(softime_avx);
	softmax_avx(input.data(), output.data(), K);
	TIMERSTOP(softime_avx);
	
	// print the results on the standard output
	if (print) {
		printResult(output, K);
	}
}

