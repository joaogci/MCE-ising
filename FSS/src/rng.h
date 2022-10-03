
#ifndef RNG_H
#define RNG_H

#include <stdint.h>

class RNG {
    private:
        uint64_t s[4];

        static inline uint64_t rotl(const uint64_t x, int k) {
            return (x << k) | (x >> (64 - k));
        }

    public:
        RNG(unsigned int seed) {
            std::srand(seed);
            for (int i = 0; i < 4; ++i)
                this->s[i] = std::rand();
        }
        ~RNG() {}

        long double rand_uniform() {
            const uint64_t result = this->rotl(this->s[0] + this->s[3], 23) + this->s[0];
            const uint64_t t = this->s[1] << 17;
            this->s[2] ^= this->s[0];
            this->s[3] ^= this->s[1];
            this->s[1] ^= this->s[2];
            this->s[0] ^= this->s[3];
            this->s[2] ^= t;
            this->s[3] = this->rotl(this->s[3], 45);
            return ((long double) result / (long double) UINT64_MAX);
        }

        uint64_t rand() {
            const uint64_t result = this->rotl(this->s[0] + this->s[3], 23) + this->s[0];
            const uint64_t t = this->s[1] << 17;
            this->s[2] ^= this->s[0];
            this->s[3] ^= this->s[1];
            this->s[1] ^= this->s[2];
            this->s[0] ^= this->s[3];
            this->s[2] ^= t;
            this->s[3] = this->rotl(this->s[3], 45);
            return result;
        }
};

#endif
