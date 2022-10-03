#include <chrono>
#include <cmath>
#include <string>

#include "src/system.h"
#include "src/fss.h"
#include "src/rng.h"

using std::string;
using std::to_string;

#define SEED ((unsigned int) time(NULL))

int L = 8;
int Sz = 2; // always Sz = 1/2
string lattice = "SS";
string dir = "./";

long REP = 1e3;
int skip = 100;

string save_name = "JDOS_L" + to_string(L) + "_" + lattice + "_npos" + to_string(Sz) + "_R1E" + to_string((int) log10(REP)) + ".txt";
string save_dir = "../JDOS/";

bool debug = true;

int main(int argv, char **argc) {
    RNG rng(SEED);
    System ising_lattice(L, Sz, lattice, dir);

    FSS fss(REP, skip, rng, ising_lattice);

    fss.simulate(0, debug);
    fss.write_to_file(save_name, save_dir, debug);
    
    return 0;
}
