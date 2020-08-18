#include "archlab.hpp"
#include <cstdlib>
#include <getopt.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "CNN/canela.hpp"
#include <opt_cnn.hpp>
#include<map>
#include<math.h>

extern model_t * build_model(const dataset_t & ds);
void run_canary(int clock_rate);
extern void train_model(model_t * model, dataset_t & train, int epochs, int batch_size=2);
extern double test_model(model_t * model, dataset_t & test, int batch_size=2);
extern double compare_model(model_t * model, model_t * batch_model, dataset_t & test, int batch_size=2);

using namespace std;

typedef std::pair<int, string> rep_map_key;
std::map<rep_map_key,float> rep_map;

void initiailize_rep_map(bool unity) {
	
#define REP_MAP_ENTRY(layer, function, REPS) do {			  \
		rep_map[rep_map_key(layer, function)] = (unity ? 1 : REPS); \
		/*std::cerr << "Setting <" << layer << ", " << function << "> = " << (unity ? 1 : REPS) << "\n";*/ \
	} while(0)
	
	// THis is a table of how many iterations of each function
	// execute in 1 second for the unoptimized version.
	REP_MAP_ENTRY(0, "activate",1.24);
	REP_MAP_ENTRY(0, "calc_grads",1.02);
	REP_MAP_ENTRY(0, "fix_weights",0.00192);
	REP_MAP_ENTRY(1, "activate",0.00475);
	REP_MAP_ENTRY(1, "calc_grads",0.0144);
	REP_MAP_ENTRY(1, "fix_weights",0);
	REP_MAP_ENTRY(2, "activate",0.0012);
	REP_MAP_ENTRY(2, "calc_grads",0.000297);
	REP_MAP_ENTRY(2, "fix_weights",0);
	REP_MAP_ENTRY(3, "activate",4.76);
	REP_MAP_ENTRY(3, "calc_grads",4.81);
	REP_MAP_ENTRY(3, "fix_weights",0.0342);
	REP_MAP_ENTRY(4, "activate",0.00289);
	REP_MAP_ENTRY(4, "calc_grads",0.00863);
	REP_MAP_ENTRY(4, "fix_weights",0);
	REP_MAP_ENTRY(5, "activate",0.000685);
	REP_MAP_ENTRY(5, "calc_grads",0.000182);
	REP_MAP_ENTRY(5, "fix_weights",0.0000000954);
	REP_MAP_ENTRY(6, "activate",1.67);
	REP_MAP_ENTRY(6, "calc_grads",1.68);
	REP_MAP_ENTRY(6, "fix_weights",0.0478);
	REP_MAP_ENTRY(7, "activate",0.00106);
	REP_MAP_ENTRY(7, "calc_grads",0.000352);
	REP_MAP_ENTRY(7, "fix_weights",0);
	REP_MAP_ENTRY(8, "activate",2.52);
	REP_MAP_ENTRY(8, "calc_grads",2.53);
	REP_MAP_ENTRY(8, "fix_weights",0.077);
	REP_MAP_ENTRY(9, "activate",0.00109);
	REP_MAP_ENTRY(9, "calc_grads",0.000335);
	REP_MAP_ENTRY(9, "fix_weights",0);
	REP_MAP_ENTRY(10, "activate",1.67);
	REP_MAP_ENTRY(10, "calc_grads",1.6);
	REP_MAP_ENTRY(10, "fix_weights",0.0506);
	REP_MAP_ENTRY(11, "activate",0.000717);
	REP_MAP_ENTRY(11, "calc_grads",0.000171);
	REP_MAP_ENTRY(11, "fix_weights",0);
	REP_MAP_ENTRY(12, "activate",0.000787);
	REP_MAP_ENTRY(12, "calc_grads",0.00207);
	REP_MAP_ENTRY(12, "fix_weights",0);
	REP_MAP_ENTRY(13, "activate",0.000206);
	REP_MAP_ENTRY(13, "calc_grads",0.0000572);
	REP_MAP_ENTRY(13, "fix_weights",0);
	REP_MAP_ENTRY(14, "activate",3.01);
	REP_MAP_ENTRY(14, "calc_grads",2.87);
	REP_MAP_ENTRY(14, "fix_weights",1.16);
	REP_MAP_ENTRY(15, "activate",3.03);
	REP_MAP_ENTRY(15, "calc_grads",2.84);
	REP_MAP_ENTRY(15, "fix_weights",1.27);
	REP_MAP_ENTRY(16, "activate",0.419);
	REP_MAP_ENTRY(16, "calc_grads",0.36);
	REP_MAP_ENTRY(16, "fix_weights",0.325);
}

uint *openmp_threads_example() {
	uint * array = new uint[1024*1024*1024];

	for (uint j = 0; j < 3; j++) {
#pragma omp parallel for 
		for(uint i= 0 ; i < 1024*1024*1024; i++) {
			array[i]+= i*j;
		}
	}
	
	return array;
}
uint *openmp_threads_simd_example() {
	uint * array = new uint[1024*1024*1024];

	for (uint j = 0; j < 3; j++) {
#pragma omp parallel for  simd
		for(uint i= 0 ; i < 1024*1024*1024; i++) {
			array[i]+= i*j;
		}
	}
	
	return array;
}
uint *openmp_simd_example() {
	uint * array = new uint[1024*1024*1024];

	for (uint j = 0; j < 3; j++) {
#pragma omp simd
		for(uint i= 0 ; i < 1024*1024*1024; i++) {
			array[i] += i*j;
		}
	}
	
	return array;
}
uint *openmp_nothing_example() {
	uint * array = new uint[1024*1024*1024];

	for (uint j = 0; j < 3; j++) {
		for(uint i= 0 ; i < 1024*1024*1024; i++) {
			array[i] += i*j;
		}
	}
	
	return array;
}

uint *gcc_simd_example() {
	typedef uint v8ui __attribute__ ((vector_size (32)));
	
	uint * array = (uint*)aligned_alloc(32, 1024*1024*1024*sizeof(uint));
	assert(sizeof(uint)==4);
	for (uint j = 0; j < 3; j++) {
		for(uint i= 0 ; i < 1024*1024*1024; i+=8) {
			v8ui *v = (v8ui*)&array[i];
			v8ui t = {(i)*j,
				  (i+1)*j,
				  (i+2)*j,
				  (i+3)*j,
				  (i+4)*j,
				  (i+5)*j,
				  (i+6)*j,
				  (i+7)*j};
		 
			*v += t; //array[i]+= i*j;
		}
	}
	
	return array;
}

void run_canary(int clock_rate)
{
	pristine_machine();
	set_cpu_clock_frequency(clock_rate);
	theDataCollector->disable_prefetcher();
	ArchLabTimer timer; // create it.
	timer.attr("function", "_canary");
	timer.go();
	archlab_canary(100000000);
}

int tag_run(ArchLabTimer & timer,
	    const std::string & kind,
	    const std::string & full_name,
	    int layer_index,
	    const std::string & function,
	    int scale) {
	timer.attr("layer_type", kind);
	timer.attr("full_name", full_name);
	timer.attr("layer", layer_index);
	timer.attr("function", function);
	//std::cout << "Timing " << full_name << "\n";
	//std::cout << "map: " << "<" << layer_index << ", " << function << "> == " <<  rep_map[rep_map_key(layer_index, function)] << "\n";
	int reps = std::min(ceil((scale + 0.0) / rep_map[rep_map_key(layer_index, function)]), 10000.0);
	timer.attr("reps", reps);
	//std::cout << "reps: " << reps << "\n";
	
	return reps;
}

	     
void time_activate(layer_t * l, int scale, const std::string& full_name, int layer_index) {
	ArchLabTimer timer; // create it.
	int reps = tag_run(timer, l->kind_str(), full_name, layer_index, "activate", scale);
	tensor_t<double> _in(l->in.size);
	timer.go();
	for (int i = 0; i < reps; i++)
		l->activate(_in);
}

void time_calc_grads(layer_t * l, int scale, const std::string& full_name, int layer_index) {
	ArchLabTimer timer; // create it.
	int reps = tag_run(timer, l->kind_str(), full_name, layer_index, "calc_grads", scale);
	tensor_t<double> _out(l->out.size);
	timer.go();
	for (int i = 0; i < reps; i++)
		l->calc_grads(_out);
}

void time_fix_weights(layer_t * l, int scale, const std::string& full_name, int layer_index) {
	ArchLabTimer timer; // create it.
	int reps = tag_run(timer, l->kind_str(), full_name, layer_index, "fix_weights", scale);
	timer.go();
	for (int i = 0; i < reps; i++)
		l->fix_weights();
}


void run_examples(int mhz)
{
	{
		ArchLabTimer timer;
		pristine_machine();
		set_cpu_clock_frequency(mhz);
		theDataCollector->disable_prefetcher();
		timer.attr("function", "openmp_threads_example");
		timer.go();
		uint * t = openmp_threads_example();
		delete t;
	}
	{
		ArchLabTimer timer;
		pristine_machine();
		set_cpu_clock_frequency(mhz);
		theDataCollector->disable_prefetcher();
		timer.attr("function", "openmp_nothing_example");
		timer.go();
		uint * t = openmp_nothing_example();
		delete t;
	}

	{
		ArchLabTimer timer;
		pristine_machine();
		set_cpu_clock_frequency(mhz);
		theDataCollector->disable_prefetcher();
		timer.attr("function", "openmp_simd_example");
		timer.go();
		uint * t = openmp_simd_example();
		delete t;
	}
	{
		ArchLabTimer timer;
		pristine_machine();
		set_cpu_clock_frequency(mhz);
		theDataCollector->disable_prefetcher();
		timer.attr("function", "openmp_threads_simd_example");
		timer.go();
		uint * t = openmp_threads_simd_example();
		delete t;
	}

	{
		ArchLabTimer timer;
		pristine_machine();
		set_cpu_clock_frequency(mhz);
		theDataCollector->disable_prefetcher();
		timer.attr("function", "gcc_simd_example");
		timer.go();
		uint * t = gcc_simd_example();
		delete t;
	}

}
	
int main(int argc, char *argv[])
{

	// Parse the command line.

	// They can select datasets and input sizes via the command
	// line in addition to all the performanec counter stuff
	std::vector<int> mhz_s;
	std::vector<int> default_mhz;
	int per_function_reps = 1;
	int train_reps = 1;
	load_frequencies();
	default_mhz.push_back(cpu_frequencies_array[0]);
	std::stringstream clocks;
	for(int i =0; cpu_frequencies_array[i] != 0; i++) {
		clocks << cpu_frequencies_array[i] << " ";
	}
	bool describe_model;
	archlab_add_flag("describe-model", describe_model, false ,  "Just print out information about the model and exit.");

	bool noscaling = false;
	archlab_add_flag("no-scale-reps", noscaling, false ,  "Don't use per-function scaling.  Just use reps directly");
	
	std::stringstream fastest;
	fastest << cpu_frequencies_array[0];
	archlab_add_option<std::vector<int> >("MHz",
					      mhz_s,
					      default_mhz,
					      fastest.str(),
					      "Which clock rate to run.  Possibilities on this machine are: " + clocks.str());
	archlab_add_option<int>("reps",
				per_function_reps,
				3,
				"3",
				"How many reps of the per-function perf tests to run.");

	archlab_add_option<int>("train-reps",
				train_reps,
				1,
				"1",
				"How many reps of the trainiing test to run");

	int scale_factor;
	archlab_add_option<int>("scale", scale_factor, 10, "The scale factor.  Bigger (smaller) numbers mean longer (shorter) run times by running more samples.  The default is 10, which should allow optimized code to run to completion without timing out.  If you want to run without opts, turn it down.");

	std::vector<std::string> dataset_s;
	std::vector<std::string> default_datasets;
	default_datasets.push_back("mininet");
	archlab_add_option<std::vector<std::string> >("dataset",
						      dataset_s,
						      default_datasets,
						      "mnist",
						      "Which dataset to use: 'mnist', 'emnist', 'cifar10', 'cifar100', or 'imagenet'. "
						      "Pass it multiple times to run multiple datasets.");


	std::vector<int> layers_to_test;
	std::vector<int> default_layers_to_test;
	layers_to_test.push_back(-1);
	archlab_add_option<std::vector<int> >("test-layer",
					      layers_to_test,
					      default_layers_to_test,
					      "0",
					      "Which layers to test.  Pass an integer corresponding to the layers order.  Pass it multiple times to run functions from multiple layers.");


	std::vector<std::string> function_s;
	std::vector<std::string> default_functions;
	default_functions.push_back("all");
	archlab_add_option<std::vector<std::string> >("function",
						      function_s,
						      default_functions,
						      "train_model",
						      "Which function to use: 'examples', 'train_model','activate', 'calc_grads', and 'fix_weights'.  Pass it multiple times to run multiple functions.  For 'calc_grads' 'fix_weights' and 'activate', it will test it on all the layers specified with '--test-layer'");
	
	archlab_parse_cmd_line(&argc, argv);

	initiailize_rep_map(noscaling);	

	
	for(auto & ds: dataset_s) {
		std::cout << "Using dataset " << ds << "\n";

			
		dataset_t *train = new dataset_t;
		//dataset_t *test = new dataset_t;
	
		if (ds == "mnist") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-train.dataset", 200 * scale_factor);
			//*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/mnist-test.dataset", 200 * scale_factor);
		} else if (ds == "emnist") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-train.dataset", 200 * scale_factor);
			//*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/mnist/emnist-byclass-test.dataset", 200 * scale_factor);
		} else if (ds == "cifar10") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar10_data_batch_1.dataset", 100 * scale_factor);
			//*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar10_test_batch.dataset", 100 * scale_factor);
		} else if (ds == "cifar100") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar100_train.dataset", 100 * scale_factor);
			//*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/cifar/cifar100_test.dataset", 100 * scale_factor);
		} else if (ds == "imagenet") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 1 * scale_factor);
			//*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 1 * scale_factor);
		} else if (ds == "mininet") {
			*train = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/mininet.dataset", 1 * scale_factor);
			//*test = dataset_t::read(std::string(std::getenv("CANELA_ROOT")) + "/datasets/imagenet/imagenet.dataset", 1 * scale_factor);
		}

		theDataCollector->register_tag("dataset", ds);
		//theDataCollector->register_tag("training_inputs_count", train->test_cases.size());
		//theDataCollector->register_tag("test_inputs_count", test->test_cases.size());
		int batch_size = 4;
		int clock_rate = mhz_s[0];
		
		run_canary(clock_rate);

		model_t * model = build_model(*train);
		model->change_batch_size(batch_size);
		
		std::cout << "Regression parameters:\n" <<model->regression_code() << std::endl;

		std::cout << "Model information:\n";
		std::cout << model->geometry() << "\n"; // output a summary of its sturcture and size.
		std::cout << "Training data size: " << (train->get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
		std::cout << "Training cases    : " << train->size() << std::endl;
		//std::cout << "Testing data size : " << (test->get_total_memory_size()+0.0)/(1024*1024)  << " MB" << std::endl;
		//std::cout << "Testing cases     : " << test->size() << std::endl;
		

		// Timing occurs inside here

		if (describe_model) {
			exit(0);
		}
		if (layers_to_test[0] == -1) {
			layers_to_test.clear();
			for (uint i = 0; i <  model->layers.size(); i++) {
				layers_to_test.push_back(i);
			}
		}
		
		for(auto & f: function_s) {
			int ran = 0;
			if (f == "examples") {
				run_examples(clock_rate);
				ran++;
			}
			if (f == "train_model"|| f == "all") {
				std::cout << "Running " << "train_model" <<"\n";
				train_model(model, *train, train_reps, batch_size);
				ran++;
			}
			
			for (int layer_index: layers_to_test) {
				if (layer_index < 0 || layer_index >= (int)model->layers.size()) {
					std::cerr << "Illegal layer index: " << layer_index << "\n";
					exit(1);
				}
								
				if (f == "activate" || f == "all") {
					std::stringstream label;
					std::cout << "Running " << "activate" << " on " << layer_index <<"\n";
					pristine_machine();
					set_cpu_clock_frequency(clock_rate);
					theDataCollector->disable_prefetcher();
					label  << "layer[" << layer_index << "]->activate " << model->layers[layer_index]->kind_str();
					time_activate(model->layers[layer_index], per_function_reps, label.str(), layer_index);
					ran++;
				}
				if (f == "calc_grads"|| f == "all") {
					std::stringstream label;
					std::cout << "Running " << "calc_grads" << " on " << layer_index <<"\n";
					pristine_machine();
					set_cpu_clock_frequency(clock_rate);
					theDataCollector->disable_prefetcher();
					label  << "layer[" << layer_index << "]->calc_grads " << model->layers[layer_index]->kind_str();
					time_calc_grads(model->layers[layer_index],per_function_reps, label.str(), layer_index);
					ran++;
				}
				if (f == "fix_weights"|| f == "all") {
					std::stringstream label;
					std::cout << "Running " << "fix_weights" << " on " << layer_index <<"\n";
					pristine_machine();
					set_cpu_clock_frequency(clock_rate);
					theDataCollector->disable_prefetcher();
					label  << "layer[" << layer_index << "]->fix_weight " << model->layers[layer_index]->kind_str();
					time_fix_weights(model->layers[layer_index], per_function_reps, label.str(), layer_index);
					ran++;
				}

			}
			if (ran == 0) {
				std::cerr << "unknown functions: " << f << "\n";
				exit(1);
			}
		}
		delete train;
	}
	archlab_write_stats();
	return 0;
}

model_t * build_model(const dataset_t & ds)  {

	// This is a simple, 1-layer model.
	
	// we use ds to set size of the inputs and the outputs of the
	// first and last layer.  In this case, they are the same
	// layer.

	//fc_layer_t *layer2 = new opt_fc_layer_t(ds.data_size, ds.label_size.x);
	// Layers build up sequentially.
	// You have to start with the
	// input layer and finish with the
	// output.
	//model->add_layer(*layer2 );
	
	model_t * model = new model_t();	

	conv_layer_t*    layer1   = new opt_conv_layer_t    ( 4, 11, 96, 0, ds.data_size);
	pool_layer_t*    layer2   = new opt_pool_layer_t    ( 2, 3, 0, layer1->out.size );	
	relu_layer_t*    layer3   = new opt_relu_layer_t    ( layer2->out.size );
			         		
	conv_layer_t*    layer4   = new opt_conv_layer_t    ( 1, 5, 256, 2, layer3->out.size );
	pool_layer_t*    layer5   = new opt_pool_layer_t    ( 2, 3, 0, layer4->out.size );	
	relu_layer_t*    layer6   = new opt_relu_layer_t    ( layer5->out.size );
			         		
	conv_layer_t*    layer7   = new opt_conv_layer_t    ( 1, 3, 384, 1, layer6->out.size );
	relu_layer_t*    layer8   = new opt_relu_layer_t    ( layer7->out.size );
			         
	conv_layer_t*    layer9   = new opt_conv_layer_t    ( 1, 3, 384, 1, layer8->out.size );
	relu_layer_t*    layer10  = new opt_relu_layer_t    ( layer9->out.size );
			         
	conv_layer_t*    layer9b  = new opt_conv_layer_t    ( 1, 3, 256, 1, layer10->out.size );
	relu_layer_t*    layer10b = new opt_relu_layer_t    ( layer9b->out.size );
			         		
	pool_layer_t*    layer11  = new opt_pool_layer_t    ( 2, 3, 0, layer10b->out.size );	
	relu_layer_t*    layer12  = new opt_relu_layer_t    ( layer11->out.size );
			         
	fc_layer_t*      layer13  = new opt_fc_layer_t        ( layer12->out.size, 4096 );
	//	dropout_layer_t *layer14  = new dropout_layer_t       (layer13->out.size, 0.5);
	fc_layer_t*      layer15  = new opt_fc_layer_t        ( layer13->out.size, 4096 );
	//	dropout_layer_t *layer16  = new dropout_layer_t       (layer15->out.size, 0.5);
	fc_layer_t*      layer17  = new opt_fc_layer_t        ( layer15->out.size, ds.label_size.x );
	//softmax_layer_t layer18(layer17.out.size);
	
	model->add_layer(*layer1 );
	model->add_layer(*layer2 );
	model->add_layer(*layer3 );
	model->add_layer(*layer4 );
	model->add_layer(*layer5 );
	model->add_layer(*layer6 );
	model->add_layer(*layer7 );
	model->add_layer(*layer8 );
	model->add_layer(*layer9 );
	model->add_layer(*layer10 );
	model->add_layer(*layer9b );
	model->add_layer(*layer10b);
	model->add_layer(*layer11 );
	model->add_layer(*layer12 );
	model->add_layer(*layer13 );
	//	model->add_layer(*layer14 );
	model->add_layer(*layer15 );
	//model->add_layer(*layer16 );
	model->add_layer(*layer17 );

	return model;
}


void train_model(model_t * model,
		 dataset_t & train,
		 int reps,
		 int batch_size) {
	int batch_index = 0;
	tensor_t<double> batch_data(tdsize(train.data_size.x, train.data_size.y, train.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(train.label_size.x, train.label_size.y, train.label_size.z, batch_size));
	for (auto& t : train.test_cases ) {
		for (int x = 0; x < t.data.size.x; x += 1)
			for (int y = 0; y < t.data.size.y; y += 1)
				for (int z = 0; z < t.data.size.z; z += 1)
					batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
		for (int x = 0; x < t.label.size.x; x += 1)
			for (int y = 0; y < t.label.size.y; y += 1)
				for (int z = 0; z < t.label.size.z; z += 1)
					batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
		batch_index += 1;
		if (batch_index >= batch_size) {
			batch_index = 0;
			{
				ArchLabTimer timer; // create it.
				pristine_machine();
				theDataCollector->disable_prefetcher();
				timer.attr("function", "train_model");
				timer.attr("reps", reps);
				timer.go();
				for (int i = 0; i < reps; i += 1) {
					std::cerr << "rep " << i << std::endl;
					model->train(batch_data, batch_label);
				}
				return;
			}
		}
	}
}

void train_model_full(model_t * model,
		 dataset_t & train,
		 int epochs,
		 int batch_size) {
	int batch_index = 0;
	tensor_t<double> batch_data(tdsize(train.data_size.x, train.data_size.y, train.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(train.label_size.x, train.label_size.y, train.label_size.z, batch_size));
	for (int i = 0; i < epochs; i += 1) {
		for (auto& t : train.test_cases ) {
			for (int x = 0; x < t.data.size.x; x += 1)
				for (int y = 0; y < t.data.size.y; y += 1)
					for (int z = 0; z < t.data.size.z; z += 1)
						batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
			for (int x = 0; x < t.label.size.x; x += 1)
				for (int y = 0; y < t.label.size.y; y += 1)
					for (int z = 0; z < t.label.size.z; z += 1)
						batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
			batch_index += 1;
			if (batch_index >= batch_size) {
				batch_index = 0;
				model->train(batch_data, batch_label);
			}
		}
	}

}

double test_model(model_t * model,
	dataset_t & test,
	int batch_size) {
	int correct = 0;
	int incorrect = 0;
	int batch_index = 0;

	tensor_t<double> batch_data(tdsize(test.data_size.x, test.data_size.y, test.data_size.z, batch_size));
	tensor_t<double> batch_label(tdsize(test.label_size.x, test.label_size.y, test.label_size.z, batch_size));
	for (auto& t : test.test_cases ) {
		for (int x = 0; x < t.data.size.x; x += 1)
			for (int y = 0; y < t.data.size.y; y += 1)
				for (int z = 0; z < t.data.size.z; z += 1)
					batch_data(x, y, z, batch_index) = t.data(x, y, z); 		
		for (int x = 0; x < t.label.size.x; x += 1)
			for (int y = 0; y < t.label.size.y; y += 1)
				for (int z = 0; z < t.label.size.z; z += 1)
					batch_label(x, y, z, batch_index) = t.label(x, y, z); 		
		batch_index += 1;
		if (batch_index >= batch_size) {
			batch_index = 0;
			tensor_t<double>& out = model->apply(batch_data);
			std::vector<tdsize> maxes = out.argmax_b();
			std::vector<tdsize> correct_maxes = batch_label.argmax_b();
			for (int i = 0; i < batch_size; i += 1) {
				if (maxes[i].x == correct_maxes[i].x) {
					correct += 1;
				} else {
					incorrect += 1;
				}
			}

		}
	}
	double total_error = (correct+0.0)/(correct+ incorrect +0.0);
	return total_error;
}


