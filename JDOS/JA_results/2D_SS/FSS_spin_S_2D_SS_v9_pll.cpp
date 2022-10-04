// v1 - from Matlab v6 (works only for spin 1/2)
// v2 - debugging for spin > 1/2 (runs for spin > 1/2, not exact)
// v3 - fixing for spin > 1/2 (done)
// v4 - cleanup + QoL
// v5 - DOS temp file saving; _0 NN list; fix final time; failsafe warning writing on output_file
// v6 - JDOS_from_contrib 
// v7 - cleanup; JDOS_contrib temp file saving; JDOS only from contrib
// v8 - using function for parallelization
// v9 - cleanup

#include <stdint.h>
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <fstream>

using namespace std;
using namespace arma;

const uint L = 4;
const uint N_atm = pow(L,2);
const uint NN = 4;
const uint N_proj = 8;
const uint REP = 1E4;
const uint skip = 1E2; // 1E2
const uint n_cores = 16;
const string E_list_filename = "E_list_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + ".dat";
const string M_list_filename = "M_list_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + ".dat";

void myFunction(imat NN_table, ivec M_list, ivec E_list, mat JDOS, uint q, mat contrib_JDOS_backup, mat* add_contrib_JDOS) {

	uint i;
	int E_old;
	uint index_E_old;
	uint q_flip;
	ivec index_spin_position1(1, fill::zeros);
	int temp_sum;
	int E_temp_subtract;
	int E_temp_add;
	uint index_NN;
	uvec hist_E(E_list.n_elem, fill::zeros);
	uvec hist_E_selected(E_list.n_elem, fill::zeros);
	int spin_step;
	int E_new;
	uint index_E_new;
	uint k;
	uint k_skip;
	ivec spin_step_list(N_proj-1, fill::zeros);
	ivec spin_step_list_index;
	ivec index_spin_position2(1, fill::zeros);
	double temp1;
	colvec temp2(2, fill::ones);
	uint i2;

	ivec SPL(N_proj, fill::zeros);
    for (i = 0; i < SPL.n_elem; i++) {
    	SPL[i] = (-1.0*N_proj+1) + 2*i;
    }

	ivec spin_vector(N_atm, fill::zeros);
	for (i = 0; i < N_atm; i++) {
		spin_vector[i] = SPL[0];
	}

	ivec SPC(N_proj, fill::zeros);
	SPC[0,0] = N_atm;

	ivec SPV(N_atm, fill::zeros);

	ivec SIM(N_atm, fill::zeros);
	for (i = 0; i < N_atm; i++) {
		SIM[i] = i;
	}

	// to reach a configuration with the desired M value

	E_old = E_list[0];
	index_E_old = 0;

	for (q_flip = 0; q_flip < q; q_flip++) {

		temp_sum = 0;
		for (i = 0; i < (N_proj-1); i++) {
			temp_sum = temp_sum + SPC[i];
		}

		index_spin_position1 = randi(1,distr_param(1, temp_sum)) - 1;

		// calculate new Energy value
    	E_temp_subtract = 0;
    	E_temp_add = 0;

		for (index_NN = 0; index_NN < NN; index_NN++) {

        	E_temp_subtract = E_temp_subtract - spin_vector(SIM(index_spin_position1[0])) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));
        	E_temp_add = E_temp_add - (spin_vector(SIM(index_spin_position1[0])) + 2) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));

    	}

    	E_old = E_old - E_temp_subtract + E_temp_add;
    	index_E_old = (E_old-E_list[0])/4;

    	// update spin_vector
    	spin_vector(SIM(index_spin_position1[0])) = spin_vector(SIM(index_spin_position1[0])) + 2;

    	// update Spin Position Count
    	SPC(SPV(SIM(index_spin_position1[0]))) = SPC(SPV(SIM(index_spin_position1[0]))) - 1;
    	SPC(SPV(SIM(index_spin_position1[0]))+1) = SPC(SPV(SIM(index_spin_position1[0]))+1) + 1;

    	// increase position by 1 of chosen spin
    	SPV(SIM(index_spin_position1[0])) = SPV(SIM(index_spin_position1[0])) + 1;
    	
    	// update Spin Index Monitor
    	temp_sum = 0;
		for(i = 0; i < SPV(SIM(index_spin_position1[0])); i++) {
			temp_sum = temp_sum + SPC[i];
		}
    	swap(SIM(index_spin_position1[0]), SIM(temp_sum));

	}

	hist_E.zeros();
	hist_E(index_E_old) = 1;

	hist_E_selected.zeros();
	hist_E_selected(index_E_old) = 1;

	mat contrib_JDOS(E_list.n_elem, N_proj-1, fill::zeros);

	// at this magnetization value, scan with all possible increases in spin position

	for (spin_step = 1; spin_step <= (N_proj-1); spin_step++) { // <--------------

		temp_sum = 0;
		for (i = 0; i < (N_proj-spin_step); i++) {
			temp_sum = temp_sum + SPC[i];
		}

		for (index_spin_position1[0] = 0; index_spin_position1[0] < (temp_sum); index_spin_position1[0]++) { // <------------

			E_temp_subtract = 0;
    		E_temp_add = 0;

			for (index_NN = 0; index_NN < NN; index_NN++) {

        		E_temp_subtract = E_temp_subtract - spin_vector(SIM(index_spin_position1[0])) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));
        		E_temp_add = E_temp_add - (spin_vector(SIM(index_spin_position1[0])) + spin_step*2) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));

    		}

    		E_new = E_old - E_temp_subtract + E_temp_add;
    		index_E_new = (E_new-E_list[0])/4;

    		// contribution to JDOS
			contrib_JDOS(index_E_new, spin_step-1) = contrib_JDOS(index_E_new, spin_step-1) + JDOS(index_E_old,q);

		}

	}

	// random walk at constant M, with random change in spin position

	k = 1;
	k_skip = 0;

	while (sum(hist_E_selected) < accu(hist_E_selected != 0)*REP) {

		ivec SPC_backup(N_proj, fill::zeros);
		SPC_backup = SPC;
		ivec spin_vector_backup(N_atm, fill::zeros);
		spin_vector_backup = spin_vector;
		ivec SPV_backup(N_atm, fill::ones);
		SPV_backup = SPV;
		ivec SIM_backup(N_atm, fill::zeros);
    	SIM_backup = SIM;

    	// choose random spin
    	index_spin_position1 = randi(1,distr_param(1, N_atm)) - 1;

    	// choose new random different spin position (check if valid for spin S)
    	for (i = 0; i <= N_proj; i++) {

    		if (i < SPV(SIM(index_spin_position1[0]))) {
    			spin_step_list[i] = i;
    		}
    		if (i > SPV(SIM(index_spin_position1[0]))) {
    			spin_step_list[i-1] = i;
    		}

    	}

    	spin_step_list_index = randi(1,distr_param(1, spin_step_list.n_elem)) - 1;
    	spin_step = spin_step_list(spin_step_list_index[0]) - SPV(SIM(index_spin_position1[0]));

    	if (spin_step >= 1) {

    		// calculate new Energy value
        	E_temp_subtract = 0;
        	E_temp_add = 0;

        	for (index_NN = 0; index_NN < NN; index_NN++) {

        		E_temp_subtract = E_temp_subtract - spin_vector(SIM(index_spin_position1[0])) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));
        		E_temp_add = E_temp_add - (spin_vector(SIM(index_spin_position1[0])) + spin_step*2) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));

    		}

    		E_new = E_old - E_temp_subtract + E_temp_add;
    		index_E_new = (E_new-E_list[0])/4;

    		// update spin_vector
    		spin_vector(SIM(index_spin_position1[0])) = spin_vector(SIM(index_spin_position1[0])) + spin_step*2;

    		// update Spin Position Count
        	SPC(SPV(SIM(index_spin_position1[0]))) = SPC(SPV(SIM(index_spin_position1[0]))) - 1;
        	SPC(SPV(SIM(index_spin_position1[0]))+spin_step) = SPC(SPV(SIM(index_spin_position1[0]))+spin_step) + 1;

        	// increase position by spin_step of chosen spin
        	SPV(SIM(index_spin_position1[0])) = SPV(SIM(index_spin_position1[0])) + spin_step;
        	
        	// update Spin Index Monitor
        	for (i2 = 0; i2 < spin_step; i2++) {
	        	temp_sum = 0; // <------------------------------------------
				for(i = 0; i < SPV(SIM(index_spin_position1[0])); i++) {
					temp_sum = temp_sum + SPC[i];
				}
	        	swap(SIM(index_spin_position1[0]), SIM(temp_sum));
        	}

        	// choose random spin that can decrease spin position by the chosen spin_step
        	temp_sum = 0;
        	for(i = 0; i < spin_step; i++) {
				temp_sum = temp_sum + SPC[i];
			}

			temp_sum++;

			if (temp_sum == N_atm) {
				index_spin_position2[0] = N_atm - 1;
			}
			else {
				index_spin_position2 = randi(1,distr_param(temp_sum, N_atm)) - 1; 
			}

			// calculate new Energy value
        	E_temp_subtract = 0;
        	E_temp_add = 0;

        	for (index_NN = 0; index_NN < NN; index_NN++) {

        		E_temp_subtract = E_temp_subtract - spin_vector(SIM(index_spin_position2[0])) * spin_vector(NN_table(SIM(index_spin_position2[0]), index_NN));
        		E_temp_add = E_temp_add - (spin_vector(SIM(index_spin_position2[0])) - spin_step*2) * spin_vector(NN_table(SIM(index_spin_position2[0]), index_NN));

    		}

    		E_new = E_new - E_temp_subtract + E_temp_add;
    		index_E_new = (E_new-E_list[0])/4;

    		// update spin_vector
    		spin_vector(SIM(index_spin_position2[0])) = spin_vector(SIM(index_spin_position2[0])) - spin_step*2;

    		// update Spin Position Count
        	SPC(SPV(SIM(index_spin_position2[0]))-spin_step) = SPC(SPV(SIM(index_spin_position2[0]))-spin_step) + 1;
        	SPC(SPV(SIM(index_spin_position2[0]))) = SPC(SPV(SIM(index_spin_position2[0]))) - 1;

        	// decrease position by 1 of chosen spin
        	SPV(SIM(index_spin_position2[0])) = SPV(SIM(index_spin_position2[0])) - spin_step;

        	// update Spin Index Monitor
        	for (i2 = 0; i2 < spin_step; i2++) {
            	temp_sum = 0;
				for(i = 0; i <= SPV(SIM(index_spin_position2[0])); i++) {
					temp_sum = temp_sum + SPC[i];
				}
	        	swap(SIM(index_spin_position2[0]), SIM(temp_sum-1));
        	}

    	}

    	else { // spin_step <= -1

    		// calculate new Energy value
        	E_temp_subtract = 0;
        	E_temp_add = 0;

        	for (index_NN = 0; index_NN < NN; index_NN++) {

        		E_temp_subtract = E_temp_subtract - spin_vector(SIM(index_spin_position1[0])) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));
        		E_temp_add = E_temp_add - (spin_vector(SIM(index_spin_position1[0])) + spin_step*2) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));

    		}

    		E_new = E_old - E_temp_subtract + E_temp_add;
    		index_E_new = (E_new-E_list[0])/4;

    		// update spin_vector
            spin_vector(SIM(index_spin_position1[0])) = spin_vector(SIM(index_spin_position1[0])) + spin_step*2;
            
            // update Spin Position Count
            SPC(SPV(SIM(index_spin_position1[0]))) = SPC(SPV(SIM(index_spin_position1[0]))) - 1;
            SPC(SPV(SIM(index_spin_position1[0]))+spin_step) = SPC(SPV(SIM(index_spin_position1[0]))+spin_step) + 1;
            
            // increase position by spin_step of chosen spin
            SPV(SIM(index_spin_position1[0])) = SPV(SIM(index_spin_position1[0])) + spin_step;

            // update Spin Index Monitor
        	for (i2 = 0; i2 < -spin_step; i2++) {
	        	temp_sum = 0;
				for(i = 0; i <= SPV(SIM(index_spin_position1[0])); i++) {
					temp_sum = temp_sum + SPC[i];
				}
	        	swap(SIM(index_spin_position1[0]), SIM(temp_sum-1));
	        }

        	// choose random spin that can decrease spin position by the chosen spin_step
        	temp_sum = 0;
        	for(i = 0; i < (N_proj+spin_step); i++) {
				temp_sum = temp_sum + SPC[i];
			}

			if (temp_sum == 1) {
				index_spin_position2[0] = 0;
			}
			else {
				index_spin_position2 = randi(1,distr_param(1, temp_sum)) - 1; 
			}

			// calculate new Energy value
            E_temp_subtract = 0;
            E_temp_add = 0;
            
            for (index_NN = 0; index_NN < NN; index_NN++) {

        		E_temp_subtract = E_temp_subtract - spin_vector(SIM(index_spin_position2[0])) * spin_vector(NN_table(SIM(index_spin_position2[0]), index_NN));
        		E_temp_add = E_temp_add - (spin_vector(SIM(index_spin_position2[0])) - spin_step*2) * spin_vector(NN_table(SIM(index_spin_position2[0]), index_NN));

    		}

    		E_new = E_new - E_temp_subtract + E_temp_add;
    		index_E_new = (E_new-E_list[0])/4;

            // update spin_vector
            spin_vector(SIM(index_spin_position2[0])) = spin_vector(SIM(index_spin_position2[0])) - spin_step*2;
            
            // update Spin Position Count
            SPC(SPV(SIM(index_spin_position2[0]))-spin_step) = SPC(SPV(SIM(index_spin_position2[0]))-spin_step) + 1;
            SPC(SPV(SIM(index_spin_position2[0]))) = SPC(SPV(SIM(index_spin_position2[0]))) - 1;
            
            // decrease position by 1 of chosen spin
            SPV(SIM(index_spin_position2[0])) = SPV(SIM(index_spin_position2[0])) - spin_step;

            // update Spin Index Monitor
        	for (i2 = 0; i2 < -spin_step; i2++) {
            	temp_sum = 0;
				for(i = 0; i < SPV(SIM(index_spin_position2[0])); i++) {
					temp_sum = temp_sum + SPC[i];
				}
	        	swap(SIM(index_spin_position2[0]), SIM(temp_sum));
	        }

    	}

    	if (JDOS(index_E_new, q) == 0) {
        
        	JDOS(index_E_new, q) = JDOS(index_E_old,q) / hist_E(index_E_old);
        	cout << "FAILSAFE FOR DOS = 0" << endl;
        	// output_Handler << "FAILSAFE FOR DOS = 0" << endl;

    	}
        
        temp1 = 1.0 * JDOS(index_E_old,q) / JDOS(index_E_new,q);
    	temp2[0] = temp1;

    	if (randu() < min(temp2)) { // accept 

    		hist_E(index_E_new) = hist_E(index_E_new) + 1;
        	E_old = E_new;
        	index_E_old = index_E_new;

    	}

    	else { // reject

    		hist_E(index_E_old) = hist_E(index_E_old) + 1;
        
            SPC = SPC_backup;
            spin_vector = spin_vector_backup;
            SPV = SPV_backup;
            SIM = SIM_backup;

    	}

    	if ((hist_E_selected(index_E_old) < REP && k_skip >= skip) || hist_E_selected(index_E_old) == 0) {

    		k_skip = 0;

    		// scan
		
			for (spin_step = 1; spin_step <= (N_proj-1); spin_step++) {

				temp_sum = 0;
				for (i = 0; i < (N_proj-spin_step); i++) {
					temp_sum = temp_sum + SPC[i];
				}

				for (index_spin_position1[0] = 0; index_spin_position1[0] < (temp_sum); index_spin_position1[0]++) { // <------------

					E_temp_subtract = 0;
					E_temp_add = 0;

					for (index_NN = 0; index_NN < NN; index_NN++) {

					E_temp_subtract = E_temp_subtract - spin_vector(SIM(index_spin_position1[0])) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));
					E_temp_add = E_temp_add - (spin_vector(SIM(index_spin_position1[0])) + spin_step*2) * spin_vector(NN_table(SIM(index_spin_position1[0]), index_NN));

					}

					E_new = E_old - E_temp_subtract + E_temp_add;
					index_E_new = (E_new-E_list[0])/4;
			
					// contribution to JDOS
					contrib_JDOS(index_E_new, spin_step-1) = contrib_JDOS(index_E_new, spin_step-1) + JDOS(index_E_old,q);

				}

			}

			hist_E_selected(index_E_old) = hist_E_selected(index_E_old) + 1;

    	}

    	k++;
    	k_skip++;

	} // RW while loop

	*add_contrib_JDOS = contrib_JDOS;

}

int main() {

	cout << "Start run for:" << endl;
	cout << "L: " << L << endl;
	cout << "N_proj: " << N_proj << endl;
	cout << "REP: " << REP << " | 1E" << to_string(int(log10(REP))) << endl;
	if (skip == 0) {
		cout << "skip: " << skip << endl;
	}
	else {
		cout << "skip: " << skip << " | 1E" << to_string(int(log10(skip))) << endl;
	}

    arma_rng::set_seed(0); // set the seed to a given value
    // arma_rng::set_seed_random();  // set the seed to a random value

    string JDOS_filename;
    string run_info_filename;
    string output_filename;
    string DOS_q_filename;
    string contrib_JDOS_q_filename;

    if (skip == 0) {
		JDOS_filename = "JDOS_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_0_x" + to_string(n_cores) + ".dat";
		run_info_filename = "run_info_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_0_x" + to_string(n_cores) + ".dat";
		output_filename = "output_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_0_x" + to_string(n_cores) + ".dat";
		DOS_q_filename = "DOS_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_0_x" + to_string(n_cores) + "_q_";
		contrib_JDOS_q_filename = "contrib_JDOS_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_0_x" + to_string(n_cores) + "_q_";
	}
	else {
		JDOS_filename = "JDOS_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_1E" + to_string(int(log10(skip))) + "_x" + to_string(n_cores) + ".dat";
		run_info_filename = "run_info_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_1E" + to_string(int(log10(skip))) + "_x" + to_string(n_cores) + ".dat";
		output_filename = "output_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_1E" + to_string(int(log10(skip))) + "_x" + to_string(n_cores) + ".dat";
		DOS_q_filename = "DOS_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_1E" + to_string(int(log10(skip))) + "_x" + to_string(n_cores) + "_q_";
		contrib_JDOS_q_filename = "contrib_JDOS_L" + to_string(L) + "_Nproj_" + to_string(N_proj) + "_R1E" + to_string(int(log10(REP))) + "_skip_1E" + to_string(int(log10(skip))) + "_x" + to_string(n_cores) + "_q_";
	}

	ofstream run_info_Handler;
	run_info_Handler.open(run_info_filename);

	ofstream output_Handler;
	output_Handler.open(output_filename);

    string NN_table_filename = "neighbour_table_2D_SS_L" + to_string(L) + "_0.txt";
    imat NN_table(N_atm, NN, fill::none);
    NN_table.load(NN_table_filename, raw_ascii);

    string coefficients_filename = "coefficients_" + to_string(N_atm) + "d" + to_string(N_proj) +  ".txt";
    vec norm_factor(N_atm + 1, fill::none);
    norm_factor.load(coefficients_filename, raw_ascii);

	uint i;

	ivec M_list(N_atm*(N_proj-1) + 1, fill::none);
	for (i = 0; i < (N_atm*(N_proj-1) + 1); i++) {
		M_list[i] = -1.0*N_atm*(1.0*N_proj-1) + 2*i;
	}

	ivec E_list(pow(N_proj-1,2)*N_atm*NN/4 + 1, fill::none);
	for (i = 0; i < (pow(N_proj-1,2)*N_atm*NN/4 + 1); i++) {
		E_list[i] = -pow(N_proj-1,2)*N_atm*NN/2 + 4*i;
	}

	mat JDOS(E_list.n_elem, M_list.n_elem, fill::zeros);
	JDOS(0,0) = 1;

	colvec DOS_temp(E_list.n_elem, fill::zeros);

	DOS_temp = JDOS.col(0);
	DOS_temp.save(DOS_q_filename + to_string(0) + ".dat", raw_ascii);

	mat contrib_JDOS_backup(E_list.n_elem, N_proj-1, fill::zeros);
	mat contrib_JDOS(E_list.n_elem, N_proj-1, fill::zeros);

	uint core_counter;

	cube contrib_JDOS_cube(E_list.n_elem, N_proj-1, n_cores, fill::zeros);

	uint q_start = 0;
	uint q_max = (M_list.n_elem-1)/2 -1;

	uint q = 0;

	time_t tstart, tend;
	tstart = time(0);

	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
	cout << put_time(&tm, "%d-%m-%Y %H:%M") << " | q: " << q << "/" << q_max << endl;

	run_info_Handler << "run with parameters" << endl;
	run_info_Handler << "L: " << L << endl;
	run_info_Handler << "N_proj: " << N_proj << endl;
	run_info_Handler << "REP: " << REP << " | 1E" << to_string(int(log10(REP))) << endl;

	if (skip == 0) {
		run_info_Handler << "skip: " << skip << endl;
	}
	else {
		run_info_Handler << "skip: " << skip << " | 1E" << to_string(int(log10(skip))) << endl;
	}
	run_info_Handler << "start at" << endl;
	run_info_Handler << put_time(&tm, "%d-%m-%Y %H:%M") << endl;

    uint spin_step;

	for (q = q_start; q <= q_max; q++) {

		time_t qstart, qend; 
		qstart = time(0);

		for (spin_step = 1; spin_step < (N_proj-1); spin_step++) {

			contrib_JDOS_backup.col(spin_step-1) = contrib_JDOS.col(spin_step);

		}

		#pragma omp parallel for schedule(dynamic)
        for (core_counter = 0; core_counter < n_cores; core_counter++) {

        	myFunction(NN_table, M_list, E_list, JDOS, q, contrib_JDOS_backup, &contrib_JDOS_cube.slice(core_counter)); //, &JDOS_cube.slice(core_counter));

        }

        // build the full JDOS_contrib

        contrib_JDOS.zeros();

        for (i = 0; i < (N_proj-1); i++) {

    		for (core_counter = 0; core_counter < n_cores; core_counter++) {

    			contrib_JDOS.col(i) = contrib_JDOS.col(i) + contrib_JDOS_cube.slice(core_counter).col(i);

        	}
    	}

		contrib_JDOS = contrib_JDOS + contrib_JDOS_backup;
		contrib_JDOS.save(contrib_JDOS_q_filename + to_string(q+1) + ".dat", raw_ascii);
		JDOS.col(q+1) = contrib_JDOS.col(0) / sum(contrib_JDOS.col(0)) * norm_factor(q+1);

		DOS_temp = JDOS.col(q+1);
        DOS_temp.save(DOS_q_filename + to_string(q+1) + ".dat", raw_ascii);

		qend = time(0);
        uvec E_hits = find(JDOS.col(q)); 

        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);

        cout << put_time(&tm, "%d-%m-%Y %H:%M") << " | q: " << q+1 << "/" << q_max+1 << " | q_time: " << difftime(qend, qstart) << " | E: " << E_hits.n_elem << " | q_time/E: " << difftime(qend, qstart)/(1.0 * E_hits.n_elem) << endl; // " | steps: " << k << endl;
        output_Handler << put_time(&tm, "%d-%m-%Y %H:%M") << " | q: " << q+1 << "/" << q_max+1 << " | q_time: " << difftime(qend, qstart) << " | E: " << E_hits.n_elem << " | q_time/E: " << difftime(qend, qstart)/(1.0 * E_hits.n_elem) << endl; // " | steps: " << k << endl;

	} 

	output_Handler.close();

	tend = time(0); 
    cout << "calc time:" << difftime(tend, tstart) << " second(s)." << endl;

    t = std::time(nullptr);
    tm = *std::localtime(&t);

    run_info_Handler << "finished at" << endl;
    run_info_Handler << put_time(&tm, "%d-%m-%Y %H:%M") << endl;
    run_info_Handler.close();

	cout << JDOS.col(q_max+1) << endl;
	// cout << JDOS << endl;

	JDOS.save(JDOS_filename, raw_ascii);
	E_list.save(E_list_filename, raw_ascii);
	M_list.save(M_list_filename, raw_ascii);

}