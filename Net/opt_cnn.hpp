#pragma once
#include"CNN/canela.hpp"

#if(0)
// These three macros are useful for tracing accesses.  See
// `example/stabilize.cpp` for an example of how to use them.
#define DUMP_ACCESS(t,x,y,z,b) do {					\
		trace << t.linearize(x,y,z,b) << " "			\
		      << " "						\
		      << &t.get(x,y,z,b) << " ";			\
	} while(0)

#define END_TRACE_LINE() do {trace << "\n";}while(0)
#define OPEN_TRACE(filename)  std::fstream trace; trace.open (filename, std::fstream::out);


// This one is customized for the stabilization code.  It prints out
// the linear index and address of each tensor element that's
// accessed.
#define DUMP_ACCESSES() do {\
		DUMP_ACCESS(in, i,b,0,0); \
		DUMP_ACCESS(weights, i,n,0,0);\
		DUMP_ACCESS(activator_input, n, 0,0, b); \
		END_TRACE_LINE();			 \
	} while(0)
#else
// By default, you get these versions, which do nothing.
#define DUMP_ACCESS(t,x,y,z,b)
#define END_TRACE_LINE
#define OPEN_TRACE(filename)
#define DUMP_ACCESSES()
#endif


// This class replaces its parent classes in the implementation of the learning
// model for this lab.  If you override functions in the baseclass by
// implementing them here, then the code here will run instead of the baseclass
// code.
//
// You should copy the functions you want to optimize into these classes, and
// confirm that the correctness tests pass.  Then, you can start modifying them
// to make them faster.
//
// The source code Canela is in /course/CSE141pp-SimpleCNN/CNN
class opt_fc_layer_t : public fc_layer_t
{
public:
	opt_fc_layer_t( tdsize in_size,
			int out_size ) : fc_layer_t(in_size, out_size) {

	}
	void activate( tensor_t<double>& in ) {
		copy_input(in);

		tdsize old_size = in.size;
		tdsize old_out_size = out.size;

		// cast to correct shape
		in.size.x = old_size.x * old_size.y * old_size.z;
		in.size.y = old_size.b;
		in.size.z = 1;
		in.size.b = 1;

		out.size.x = old_out_size.x * old_out_size.y * old_out_size.z;
		out.size.y = old_out_size.b;
		out.size.z = 1;
		out.size.b = 1;

		#define TILE_SIZE 4	
		for ( int b = 0; b < activator_input.size.b; b++) {
			for ( int n = 0; n < activator_input.size.x; n++ ) {
				activator_input(n, 0, 0, b) = 0;
			}
		}
		for (int nn = 0; nn < out.size.x; nn+=TILE_SIZE){
			for ( int b = 0; b < in.size.y; b++ ) {
				//for ( int n = 0; n < out.size.x; n++ ) {
				for ( int n = nn; n < nn+TILE_SIZE && n < out.size.x; n++ ) {
					for ( int i = 0; i < in.size.x; i++ ) {
						double in_val = in(i, b, 0);
						double weight_val = weights( i, n, 0 );
						double mul_val = in_val * weight_val;
						double acc_val = activator_input(n, 0, 0, b) + mul_val;
						activator_input(n, 0, 0, b) = acc_val;
					}
				}
			}
		}

		// finally, apply the activator function.
		for ( unsigned int n = 0; n < activator_input.element_count(); n++ ) {
			out.data[n] = activator_function( activator_input.data[n] );
		}

		// don't forget to reset the shapes
		in.size = old_size;
		out.size = old_out_size;
	}

	void calc_grads( const tensor_t<double>& grad_next_layer ) {
		
		memset( grads_out.data, 0, grads_out.size.x * grads_out.size.y * grads_out.size.z * sizeof( double ) );

                grads_out.size.x = grads_out.size.x * grads_out.size.y * grads_out.size.z;
                grads_out.size.y = 1;
                grads_out.size.z = 1;
		
		for ( int b = 0; b < out.size.b; b++ ) {
                        for ( int n = 0; n < activator_input.size.x; n++ ){
				double ad = activator_derivative( activator_input(n, 0, 0, b) );
				double ng = grad_next_layer(n, 0, 0, b);
				act_grad(n, 0, 0, b) = ad * ng;
                        }
                }
		
		for (int nn = 0; nn < out.size.x; nn+=TILE_SIZE){
                	for ( int b = 0; b < out.size.b; b++ ) {
				//for ( int n = 0; n < out.size.x; n++ ) {
				for ( int n = nn; n < nn+TILE_SIZE && n < out.size.x; n++ ) {
					for ( int i = 0; i < grads_out.size.x; i++ ) {
                                        	grads_out(i, 0, 0, b) += act_grad(n, 0, 0, b) * weights( i, n, 0);
                                	}
                        	}
                	}
		}
		grads_out.size = in.size;
	}
	void fix_weights() {

		tdsize old_in_size = in.size;
		in.size.x = in.size.x * in.size.y * in.size.z;
		in.size.y = 1;
		in.size.z = 1;
		for(int nn = 0; nn < weights.size.y; nn+=TILE_SIZE){
                for ( int b = 0; b < out.size.b; b++ ) {
                //{ int b = 1;
                        for ( int n = nn; n < weights.size.y && n < nn+TILE_SIZE; n++ ) {
                                for ( int i = 0; i < weights.size.x; i++ ) {
                                        double& w = weights( i, n, 0 );
                                        double m = (act_grad(n, 0, 0, b) + old_act_grad(n, 0, 0, b) * MOMENTUM);
                                        double g_weight = w - (LEARNING_RATE * m * in(i, 0, 0, b) + LEARNING_RATE * WEIGHT_DECAY * w);
                                        w = g_weight;
                                }
                                old_act_grad(n, 0, 0, b) = act_grad(n, 0, 0, b) + old_act_grad(n, 0, 0, b) * MOMENTUM;
                        }
                }
		in.size = old_in_size;
	}}

};

class opt_conv_layer_t: public conv_layer_t
{
public:
	
	opt_conv_layer_t( uint16_t stride,
			  uint16_t kernel_size, 
			  uint16_t kernel_count,
			  double pad,
			  tdsize in_size
			  ) : conv_layer_t(stride, kernel_size, kernel_count, pad, in_size){}

	void activate( tensor_t<double>& in ) {
		copy_input(in);
		#pragma omp parallel for
		for ( int b = 0; b < out.size.b; b++ ) {
			for ( uint filter = 0; filter < filters.size(); filter++ ) {
				tensor_t<double>& filter_data = filters[filter];
				for ( int y = 0; y < out.size.y; y++ ) {
					for ( int x = 0; x < out.size.x; x++ ) {
						point_t mapped(x*stride, y*stride, 0);
						double sum = 0;
						for ( int z = 0; z < in.size.z; z++ ) {
							for ( int j = 0; j < kernel_size; j++ ) { 
								for ( int i = 0; i < kernel_size; i++ ){
									double f = filter_data( i, j, z );
								
									double v;
									if (mapped.x + i >= in.size.x ||
								    	mapped.y + j >= in.size.y) {
										v = pad;
									} else {
										v = in( mapped.x + i, mapped.y + j, z, b );
									}
									sum += f*v;
								}}}
						out( x, y, filter, b ) = sum;
					}
				}
			}
		}
	}

	void calc_grads(const tensor_t<double>& grad_next_layer ) {
		throw_assert(grad_next_layer.size == out.size, "mismatch input size for calc_grads");
		for ( uint k = 0; k < filter_grads.size(); k++ ) 
			for ( int b = 0; b < in.size.b; b++ )
				for ( int z = 0; z < in.size.z; z++ )
					for ( int j = 0; j < kernel_size; j++ )
						for ( int i = 0; i < kernel_size; i++ )
							filter_grads[k].get( i, j, z, b ).grad = 0;
	
		#pragma omp parallel for 	
		for ( int b = 0; b < in.size.b; b++ ) {
			for ( int z = 0; z < in.size.z; z++ ) {
				for ( int y = 0; y < in.size.y; y++ ) {
					for ( int x = 0; x < in.size.x; x++ ) {
						range_t rn = map_to_output( x, y );
						double sum_error = 0;
						for ( int k = rn.min_z; k <= rn.max_z; k++ ) {
							for ( int j = rn.min_y; j <= rn.max_y; j++ ) {
								int miny = j * stride;
								for ( int i = rn.min_x; i <= rn.max_x; i++ ) {
									int minx = i * stride;
									int w_applied = filters[k].get( x - minx, y - miny, z );
									sum_error += w_applied * grad_next_layer( i, j, k, b );
									filter_grads[k].get( x - minx, y - miny, z, b ).grad += in( x, y, z, b ) * grad_next_layer( i, j, k, b );
								}
							}
						}
						grads_out( x, y, z, b ) = sum_error;
					}
				}
			}
		}
	}
	
	void fix_weights() {
		#pragma omp parallel for
		for ( uint a = 0; a < filters.size(); a++ ){
			for ( int b = 0; b < in.size.b; b++ ){
				for ( int z = 0; z < in.size.z; z++ ) {
					for ( int j = 0; j < kernel_size; j++ ){
						for ( int i = 0; i < kernel_size; i++ ){
							double& w = filters[a].get( i, j, z );
							gradient_t& grad = filter_grads[a].get( i, j, z, b );
							w = update_weight( w, grad );
							update_gradient( grad );
						}
					}
				}

			}
		}
	}
};

class opt_pool_layer_t: public pool_layer_t
{
public:
	opt_pool_layer_t( uint16_t stride,
			  uint16_t filter_size,
			  double pad,
			  tdsize in_size ) : pool_layer_t(stride, filter_size, pad, in_size){}
	void activate(tensor_t<double>& in ) {
		copy_input(in);
		#pragma omp parallel for
		for ( int b = 0; b < out.size.b; b++ ) {
			for ( int z = 0; z < out.size.z; z++ ) {
				for ( int y = 0; y < out.size.y; y++ ) {
					for ( int x = 0; x < out.size.x; x++ ) {
						point_t mapped(x*stride, y*stride, 0);
						double mval = -FLT_MAX;
						for ( int i = 0; i < filter_size; i++ )
							for ( int j = 0; j < filter_size; j++ ) {
								double v;
								if (mapped.x + i >= in.size.x ||
							    	mapped.y + j >= in.size.y) {
									v = pad;
								} else {
									v = in( mapped.x + i, mapped.y + j, z );
								}

								if ( v > mval )
									mval = v;
							}
						out( x, y, z, b ) = mval;
					}
				}
			}
		}
	}

	void calc_grads(const tensor_t<double>& grad_next_layer )
	{
		#pragma omp parallel for
		for ( int b = 0; b < in.size.b; b++ ) {
			for ( int z = 0; z < in.size.z; z++ ) {
				for ( int y = 0; y < in.size.y; y++ ) {
					for ( int x = 0; x < in.size.x; x++ ) {
					range_t rn = map_to_output( x, y );
						double sum_error = 0;
						for ( int j = rn.min_y; j <= rn.max_y; j++ ) {
							for ( int i = rn.min_x; i <= rn.max_x; i++ ) {
								int is_max = in( x, y, z ) == out( i, j, z ) ? 1 : 0;
								sum_error += is_max * grad_next_layer( i, j, z );
							}
						}
						grads_out( x, y, z, b ) = sum_error;
					}
				}
			}
		}
	}
};

class opt_relu_layer_t : public relu_layer_t
{
public:
	opt_relu_layer_t(const tdsize & in_size )
		:
		relu_layer_t(in_size)
	{
	}
	void activate(tensor_t<double>& in ) {
		copy_input(in);
		#pragma omp parallel for
		for (int b = 0; b < in.size.b; b++ ){
			for ( int z = 0; z < in.size.z; z++ ){
				for ( int y = 0; y < in.size.y; y++ ){
					for ( int x = 0; x < in.size.x; x++ )
					{
						double v = in( x, y, z, b );
						if ( v < 0 ) {
							v = 0;
						}
						out( x, y, z, b ) = v;
					}
				}
			}
		}
	}

	void calc_grads(const tensor_t<double>& grad_next_layer )
	{
		throw_assert(grad_next_layer.size == in.size, "mismatched input");
		for(int b = 0; b < in.size.b; b++){
			for ( int z = 0; z < in.size.z; z++ ){
				for ( int j = 0; j < in.size.y; j++ ){
					for ( int i = 0; i < in.size.x; i++ )
					{
						grads_out( i, j, z ) = (in( i, j, z ) < 0) ?
						(0) :
						(grad_next_layer( i, j, z ));
					}
				}
			}
		}
	}
};
