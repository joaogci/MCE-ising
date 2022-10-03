/**
 * Implementation by João Inácio (j.g.c.inacio@fys.uio.no).
 * Dec. 2021
 */

#include "fss.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdint.h>
#include <filesystem>

/**
 * Flat Scan Sampling class. 
 * @param rng (RNG): Reference to an RNG object.
 * @param ising_lattice (System): reference to a System object. 
 */
FSS::FSS(RNG &rng, System &ising_lattice) {
    this->set_rng(rng);
    this->set_lattice(ising_lattice);
}

/**
 * Flat Scan Sampling class. 
 * @param rng (RNG): Reference to an RNG object.
 */
FSS::FSS(RNG &rng) {
    this->set_rng(rng);
}

/**
 * Flat Scan Sampling class. 
 * @param REP (unsigned long): Value of REP for the calculations. Should be large enough to ensure convergence.
 * @param skip (int): Value of the skip parameter. Usually set to the number of spins in the system.
 * @param rng (RNG): Reference to an RNG object.
 */
FSS::FSS(long long REP, int skip, RNG &rng) {
    this->set_rng(rng);
    this->set_params(REP, skip);
}

/**
 * Flat Scan Sampling class. 
 * @param REP (unsigned long): Value of REP for the calculations. Should be large enough to ensure convergence.
 * @param skip (int): Value of the skip parameter. Usually set to the number of spins in the system.
 * @param rng (RNG): Reference to an RNG object.
 * @param ising_lattice (System): reference to a System object. 
 */
FSS::FSS(long long REP, int skip, RNG &rng, System &ising_lattice) {
    this->set_rng(rng);
    this->set_params(REP, skip);
    this->set_lattice(ising_lattice);
}

FSS::~FSS() {
    delete[] this->hist;
    delete[] this->flip_list;
    delete[] this->time_iter;
    delete[] this->steps_iter;
}

/**
 * Sets the parameters to the Flat Scan Sampling instance.
 * @param REP (unsigned long): Value of REP for the calculations. Should be large enough to ensure convergence.
 * @param skip (int): Value of the skip parameter. Usually set to the number of spins in the system.
 */
void FSS::set_params(long long REP, int skip) {
    this->REP = REP;
    this->skip = skip;

    this->run_time = 0;
    
    this->added_params = true;
}

/**
 * Sets Ising lattice for Flat Scan Sampling class.
 * @param ising_lattice (System): Reference to System object.
 */
void FSS::set_lattice(System &ising_lattice) {
    this->ising_lattice = &(ising_lattice);
    this->ising_lattice->init_spins_random(*(this->rng));

    this->hist = new unsigned long long[this->ising_lattice->NE];
    this->flip_list = new std::vector<int>[2];

    for (int i = 0; i < this->ising_lattice->NE; ++i) {
        this->hist[i] = 0;
    }

    this->idx_M_max = (this->ising_lattice->NM + 1) / 2 - 2;
    if (this->ising_lattice->NM % 2 == 0)
        this->idx_M_max = this->ising_lattice->NM / 2 - 3;

    this->time_iter = new double[this->idx_M_max];
    this->steps_iter = new unsigned long long[this->idx_M_max];

    this->added_lattice = true;
}

/**
 * Sets RNG for the Flat Scan Sampling class.
 * @param rng (RNG): Reference to RNG object.
 */
void FSS::set_rng(RNG &rng) {
    this->rng = &(rng);

    this->added_rng = true;
}

/**
 * Estimates the Joint Density of States for with the Flat Scan Sampling
 * algorithm, with respect to the parameters and lattice passes beforehand.
 * @param run (int): Should be passed 0, if only doing one run. Used to keep track of how many times 
 * the algorithm has run in a given script.
 * @param verbose (bool): If true, output information about the simulation each iteration.
 */
void FSS::simulate(int run, bool verbose) {
    if (!(this->added_lattice && this->added_params && this->added_rng)) {
        printf(" -- Error: forgot to add the simulation parameters, rng or lattice -- ");
    }

    printf("Initiating Flat Scan Sampling Simulation; run: %d \n", run);
    time_t now = time(0);
    std::string t = ctime(&now); t.pop_back();
    printf("    Time: %s \n", t.c_str());
    if (verbose) {    
        printf("    System:  L: %d | Sz: %d | N_atm: %d | lattice: %s | NN: %d \n",
            this->ising_lattice->L, 
            this->ising_lattice->Sz,
            this->ising_lattice->N_atm,
            this->ising_lattice->lattice.c_str(), 
            this->ising_lattice->NN);
        printf("    Simulation Parameters: REP: %lld | skip: %d \n",
            this->REP, 
            this->skip);   
        printf("\n");
    }
    
        int *new_spins_vector = new int[this->ising_lattice->N_atm];

    auto runtime_start = std::chrono::steady_clock::now();

    this->first_step();
    if (verbose) {
        time_t now = time(0);
        std::string t = ctime(&now); t.pop_back();
        now = time(0);
        t = ctime(&now); t.pop_back();
        printf("%s | run: %d | q: %d/%d \n", 
            t.c_str(),
            run, 
            0, 
            this->idx_M_max);
    }

    for (int q = 1; q <= this->idx_M_max; ++q) {
        auto q_start = std::chrono::steady_clock::now();

        int idx_E_config;
        this->random_config_q(q, idx_E_config);
        this->hist[idx_E_config]++;
        this->scan(q, idx_E_config);

        long long k = 1;
        bool accepted = false;

        int new_E_config, new_idx_E_config;
        int flipped_idx1, flipped_idx2;

        long double ratio;

        while (this->min_hist() < this->REP) {
            for (int i = 0; !accepted && i < this->ising_lattice->N_atm; ++i)
                new_spins_vector[i] =  this->ising_lattice->spins_vector[i];

            this->spin_flip(new_spins_vector, new_E_config, new_idx_E_config, flipped_idx1, flipped_idx2); 

            ratio = this->ising_lattice->JDOS[idx_E_config * this->ising_lattice->NM + q] / this->ising_lattice->JDOS[new_idx_E_config * this->ising_lattice->NM + q];

            accepted = false;
            if (ratio >= 1 || this->rng->rand_uniform() < ratio || this->hist[new_idx_E_config] == 0) {
                for (int i = 0; i < this->ising_lattice->N_atm; ++i)
                    this->ising_lattice->spins_vector[i] = new_spins_vector[i];

                this->ising_lattice->E_config = new_E_config;
                idx_E_config = new_idx_E_config;
                accepted = true;
            }
            else if (flipped_idx1 != flipped_idx2) {
                this->flip_list[0].pop_back();
                this->flip_list[0].push_back(flipped_idx1);

                this->flip_list[1].pop_back();
                this->flip_list[1].push_back(flipped_idx2);
            }

            if (this->hist[idx_E_config] < this->REP && k % this->skip == 0 || this->hist[idx_E_config] == 0) {
                this->hist[idx_E_config]++;
                this->scan(q, idx_E_config);
            }

            k++;
        }

        int hits = this->normalize_JDOS(q);

        auto q_end = std::chrono::steady_clock::now();
        double q_time = (double) (std::chrono::duration_cast<std::chrono::microseconds> (q_end - q_start).count()) * pow(10, -6);

        if (verbose) {
            time_t now = time(0);
            std::string t = ctime(&now); t.pop_back();
            now = time(0);
            t = ctime(&now); t.pop_back();
            printf("%s | run: %d | q: %d/%d | time: %fs | E: %d | time/E: %fs \n", 
                t.c_str(),
                run, 
                q, 
                this->idx_M_max, 
                q_time, 
                hits, 
                q_time / hits);
        }

        this->time_iter[q - 1] = q_time;
        this->steps_iter[q - 1] = k;
    }

    auto runtime_end = std::chrono::steady_clock::now();
    this->run_time = (double) (std::chrono::duration_cast<std::chrono::microseconds> (runtime_end - runtime_start).count()) * pow(10, -6);

    delete[] new_spins_vector;

    if (verbose) {
        printf("\n");
    }
    printf("    Run time: %fs \n", this->run_time);
    now = time(0);
    t = ctime(&now); t.pop_back();
    printf("    Time: %s \n", t.c_str());
    printf("Finished Flat Scan Sampling Simulation; run : %d \n", run);
}

void FSS::first_step() {
    this->ising_lattice->JDOS[0] = 1;

    int E_tmp1, E_tmp2, E_tmp3;
    int idx_E_tmp3;

    for (int flip_idx = 0; flip_idx < this->ising_lattice->N_atm; ++flip_idx) {
        E_tmp1 = 0;
        for (int a = 0; a < this->ising_lattice->NN; ++a) {
            E_tmp1 += - 1;
        }
        E_tmp2 = 0;
        for (int a = 0; a < this->ising_lattice->NN; ++a) {
            E_tmp2 += 1;
        }
        
        E_tmp3 = - this->ising_lattice->max_E - E_tmp1 + E_tmp2;
        idx_E_tmp3 = this->ising_lattice->energies[E_tmp3];

        this->ising_lattice->JDOS[idx_E_tmp3 * this->ising_lattice->NM + 1] += this->ising_lattice->JDOS[0];
    }

    int sum_JDOS = 0;
    for (int i = 0; i < this->ising_lattice->NE; ++i) {
        if (this->ising_lattice->JDOS[i * this->ising_lattice->NM + 1] > 0) {
            sum_JDOS += this->ising_lattice->JDOS[i * this->ising_lattice->NM + 1];
        }
    }

    for (int i = 0; i < this->ising_lattice->NE; ++i) {
        this->ising_lattice->JDOS[i * this->ising_lattice->NM + 1] = this->ising_lattice->JDOS[i * this->ising_lattice->NM + 1] * exp(this->ising_lattice->norm_factor[1]) / sum_JDOS;
    }
}

void FSS::random_config_q(int &q, int &idx_E_config) {
    for (int i = 0; i < this->ising_lattice->NE; ++i) {
        this->hist[i] = 0;
    }
    for (int i = 0; i < 2; ++i) {
        this->flip_list[i].clear();
    }

    this->ising_lattice->init_spins_max_M();
    for (int i = 0; i < this->ising_lattice->N_atm; ++i) {
        this->flip_list[0].push_back(i);
    }

    int idx_tmp, flipped_idx;
    int delta_E;

    for (int idx = 1; idx <= q; ++idx) {
        idx_tmp = this->rng->rand() % flip_list[0].size();
        flipped_idx = this->flip_list[0].at(idx_tmp);
        this->ising_lattice->spins_vector[flipped_idx] = - 1;

        this->flip_list[1].push_back(flipped_idx);
        this->flip_list[0].erase(this->flip_list[0].begin() + idx_tmp);
        
        delta_E = 0;
        for (int a = 0; a < this->ising_lattice->NN; ++a) {
            delta_E += - this->ising_lattice->spins_vector[flipped_idx] * this->ising_lattice->spins_vector[this->ising_lattice->NN_table[flipped_idx * this->ising_lattice->NN + a]];
        }
        
        this->ising_lattice->E_config += 2 * delta_E;
    }

    idx_E_config = this->ising_lattice->energies[this->ising_lattice->E_config];
}

void FSS::scan(int &q, int &idx_E_config) {
    int delta_E, E_tmp;
    int idx_E_tmp;

    for (int flip_idx = 0; flip_idx < this->flip_list[0].size(); ++flip_idx) {
        delta_E = 0;
        for (int a = 0; a < this->ising_lattice->NN; ++a)
            delta_E += this->ising_lattice->spins_vector[this->flip_list[0].at(flip_idx)] * this->ising_lattice->spins_vector[this->ising_lattice->NN_table[this->flip_list[0].at(flip_idx) * this->ising_lattice->NN + a]];

        E_tmp = this->ising_lattice->E_config + 2 * delta_E;
        idx_E_tmp = this->ising_lattice->energies[E_tmp];

        this->ising_lattice->JDOS[idx_E_tmp * this->ising_lattice->NM + q + 1] += this->ising_lattice->JDOS[idx_E_config * this->ising_lattice->NM + q] / this->REP;
    }
}

void FSS::spin_flip(int *new_spins_vector, int &new_E_config, int &new_idx_E_config, int &flipped_idx1, int &flipped_idx2) {
    int idx_tmp1 = this->rng->rand() % this->flip_list[0].size();
    flipped_idx1 = this->flip_list[0].at(idx_tmp1);
    new_spins_vector[flipped_idx1] = - 1;

    this->flip_list[1].push_back(flipped_idx1);
    this->flip_list[0].erase(this->flip_list[0].begin() + idx_tmp1);

    int delta_E = 0;
    for (int a = 0; a < this->ising_lattice->NN; ++a)
        delta_E += - new_spins_vector[flipped_idx1] * new_spins_vector[this->ising_lattice->NN_table[flipped_idx1 * this->ising_lattice->NN + a]];
    new_E_config = this->ising_lattice->E_config + 2 * delta_E;

    int idx_tmp2 = this->rng->rand() % this->flip_list[1].size();
    flipped_idx2 = this->flip_list[1].at(idx_tmp2);
    new_spins_vector[flipped_idx2] = 1;

    this->flip_list[0].push_back(flipped_idx2);
    this->flip_list[1].erase(this->flip_list[1].begin() + idx_tmp2);

    delta_E = 0;
    for (int a = 0; a < this->ising_lattice->NN; ++a)
        delta_E += - new_spins_vector[flipped_idx2] * new_spins_vector[this->ising_lattice->NN_table[flipped_idx2 * this->ising_lattice->NN + a]];
    new_E_config = new_E_config + 2 * delta_E;

    new_idx_E_config = this->ising_lattice->energies[new_E_config];
}

/**
 * Writes estimated Joint Density of States to file.
 * @param name (string): Name of the file with extension.
 * @param path (string): Path of the directory where the file should be written to. 
 * Relative path from the execution directory, of the form "folder1/folder2/.../".
 * @param debug (bool): If true outputs additional information about the simulation.
 * First column is the number of steps and second is the time taken.
 */
void FSS::write_to_file(std::string name, std::string path, bool debug) {
    std::filesystem::create_directories(path); 

    std::ofstream file1(path + name);
    if (file1.is_open()) {
        for (int i = 0; i < this->ising_lattice->NE; ++i) 
        {
            for (int j = 0; j < this->ising_lattice->NM; ++j) {
                file1 << this->ising_lattice->JDOS[i * this->ising_lattice->NM + j] << " ";
            }
            file1 << "\n";
        }
        file1.close();
    } else {
        std:printf(" -- Error: can not open save file, please check you directory -- \n");
    }

    if (debug) {
        std::filesystem::create_directories(path + "debug/"); 

        std::ofstream file2(path + "debug/" + name);
        if (file2.is_open()) {
            file2 << this->run_time << "\n";
            for (int i = 0; i < this->idx_M_max; i++) {
                file2 << this->steps_iter[i] << " " << this->time_iter[i] << "\n";
            }
            file2.close();
        } else {
            printf(" -- Error: can not open debug file, please check you directory -- \n");
        }
    }
}

/**
 * Prints the estimated Joint Density of States to the terminal.
 */
void FSS::print_JDOS() {
    printf("\nJDOS: \n");
    for (int i = 0; i < this->ising_lattice->NE; ++i) 
    {
        for (int j = 0; j < this->ising_lattice->NM; ++j) {
            std::cout << this->ising_lattice->JDOS[i * this->ising_lattice->NM + j] << " ";
        }
        std::cout << std::endl;
    } 
}

int FSS::normalize_JDOS(int &q) {
    long double sum_JDOS = 0;
    for (int i = 0; i < this->ising_lattice->NE; ++i)
        if (this->ising_lattice->JDOS[i * this->ising_lattice->NM + q + 1] > 0)
            sum_JDOS += this->ising_lattice->JDOS[i * this->ising_lattice->NM + q + 1];

    for (int i = 0; i < this->ising_lattice->NE; ++i)
        this->ising_lattice->JDOS[i * this->ising_lattice->NM + q + 1] = this->ising_lattice->JDOS[i * this->ising_lattice->NM + q + 1] * exp(this->ising_lattice->norm_factor[q + 1]) / sum_JDOS;

    int hits = 0;
    for (int i = 0; i < this->ising_lattice->NE; ++i)
        if (this->ising_lattice->JDOS[i * this->ising_lattice->NM + q] > 0)
            hits++;
    return hits;
}

unsigned long long FSS::min_hist() {
    unsigned long long min = __UINT64_MAX__;
    for (int i = 0; i < this->ising_lattice->NE; i++)
        if (hist[i] != 0 && this->hist[i] < min)
            min = this->hist[i];
    return min;
}

