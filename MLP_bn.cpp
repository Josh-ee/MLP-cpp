#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <fstream>

#include <bits/stdc++.h> // use this for sort

using namespace std;

class Matrix {
private:
    vector<vector<double>> data;

public:
    // Constructors
    // constructor/init
    Matrix();

    // copy constructor
    Matrix(const Matrix& other); 

    //deconstructor
    ~Matrix();

    // Functions
    // We use alot of size_t since the vectors size can be large or small depending on topology
    // https://www.geeksforgeeks.org/size_t-data-type-c-language/
    
    // matrix.rows() returns size of rows
    size_t rows() const; 
    
    // matrix.cols() returns size of cols
    size_t cols() const; 

    //pushback  does the same thing as a vector
    void push_back(vector<double> row); 

    // resize is used to split a Matrix 
    void resize(size_t rows, size_t cols); 

    // this loops through with iterators to display the matrix
    void display() const; 

    // operator overloading
    vector<double>& operator[](size_t row); 
    const vector<double>& operator[](size_t row) const;

};

Matrix::Matrix() // constructor
{ 
    // cout << "Matrix initilized" <<endl;
}

Matrix::Matrix(const Matrix& other) // copy constructor
{ 
    // cout << "Matrix copy" <<endl;
    data = other.data;

}

Matrix::~Matrix() 
{
    // cout << "Matrix destroyed" << endl;
}



size_t Matrix::rows() const // matrix.rows() returns size of rows
{
    size_t sz;
    if (data.empty())
    {
        sz = 0;
    }
    else
    {
        sz = data.size();
    }
    return sz;
}

size_t Matrix::cols() const // matrix.cols() returns size of cols
{
    size_t sz;
    if (data.empty())
    {
        sz = 0;
    }
    else
    {
        sz = data[0].size();
    }
    return sz;

}

void Matrix::push_back(vector<double> row) 
{
    data.push_back(row);
}

// need to resize both rows and cols
void Matrix::resize(size_t rows, size_t cols)
{
    data.resize(rows);
    for (auto& row : data) 
    {
        row.resize(cols);
    }
}

void Matrix::display() const
{   
    // print rows and cols via iterators 
    for (auto row_it = data.begin(); row_it != data.end(); ++row_it) 
    {
        for (auto col_it = row_it->begin(); col_it != row_it->end(); ++col_it) 
        {
            cout << *col_it << " ";
        }
        cout << endl;
    }
}

// overload [] so we can get values with matrix[i][j]
vector<double>& Matrix::operator[](size_t row)
{ 
    return data[row]; 
}

const vector<double>& Matrix::operator[](size_t row) const
{ 
    return data[row]; 
}

// Vector Operations is used to evaluate performance 
class VectorOps
{
private: 
    vector<double> sequence;
    
    double median;
    double mode;
    double std_dev;
    double variance;

public:
    double mean;
    //constructors
    //constructor/init
    VectorOps(); 
    // copy constructor
    VectorOps(const VectorOps& other); 
    //deconstructor
    ~VectorOps(); 
    //functions
    //add is like push_back
    bool add(const double num); 
    bool calc_mean();
    bool calc_median();
    bool calc_mode();
    bool calc_std_dev_var();
    bool calc_stats();
    // bool calc_variance();
    //prints values in the vector
    void display_vect() const;
    void display_stats() const; //prints stats
    
    
};

VectorOps::VectorOps() // constructor
{ 
    sequence = {};
    mean = 0;
    mode = 0;
    std_dev = 0;
    variance = 0;
}

VectorOps::VectorOps(const VectorOps& other) // copy constructor
{ 
    sequence = other.sequence;
    mean = other.mean;
    median = other.median;
    mode = other.mode;
    std_dev = other.std_dev;
    variance = other.variance;
}
VectorOps::~VectorOps() // deconstructor
{ 
	//cout << "Vect is removed" << endl;
}

bool VectorOps::add(const double num)
{ 
    sequence.push_back(num);
    return true;
}

bool VectorOps::calc_mean()
{
    // auto size = sequence.size(); //size type
    double m=0;
    for (auto it = sequence.cbegin(); it != sequence.cend(); ++it)
    { 
        // cout << *it << " ";
        m += *it;
    }
    mean = m/sequence.size();
    return true;
}

bool VectorOps::calc_median()
{ 
    vector<double> temp = sequence;
    // sort the matrix before finding median
    sort(temp.begin(), temp.end()); 
    auto size = temp.size();
    //odd
    if (size%2)
    { 
        median = temp.at(size/2);
    }
    //even
    else
    { 
        median = (temp.at(size/2 -1) + temp.at(size/2))/2;
    }
    return true;
}

bool VectorOps::calc_mode() 
{ 
    auto beg = sequence.cbegin(); //constant itr read only
	auto end = sequence.cend(); //constant itr
    auto size = sequence.size(); //size type
    decltype(size) cur_count = 1; //make counter size type
    decltype(size) count = 1;
    double cur_mode = *beg; //derefrence to get first val
    
    
    for (decltype(size) i = 1; i < sequence.size(); ++i) 
    {
        if (sequence[i] == sequence[i-1]) 
        {
            count++;
        } 
        else 
        {
            if (count > cur_count) 
            {
                cur_count = count;
                cur_mode = sequence[i-1];
            }
            count = 1;
        }
    }

    if (count > cur_count) 
    {
        cur_count = count;
        cur_mode = sequence[sequence.size()-1];
    }
    mode = cur_mode;
    return true;
}


bool VectorOps::calc_std_dev_var()
{
    auto size = sequence.size(); //size type
    double sum = 0.0;
    for (auto it = sequence.cbegin(); it != sequence.cend(); ++it)
    { 
        sum += (*it - mean)* (*it - mean);
    }
    double var = sum / size;
    std_dev = sqrt(var);
    variance = var;
    return true;
}

bool VectorOps::calc_stats()
{
    this->calc_mean();
    this->calc_median();
    this->calc_mode();
    this->calc_std_dev_var();
    return true;
}


void VectorOps::display_vect() const
{ 
    //constant itr read only
    for (auto it = sequence.cbegin(); it != sequence.cend(); ++it)
    {   
        // dereference iter to get value
        cout << *it << " ";
    }
    cout << endl;
    return;
}

void VectorOps::display_stats() const
{ 
    cout << "Mean: " << mean << endl;
    cout << "Median: " << median << endl;
    // cout << "Mode: " << mode << endl;
    cout << "Standard Deviation: " << std_dev << endl;
    cout << "Variance: " << variance << endl;
    return;
}

// end of VectorOps

double sigmoid(double x) 
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) 
{
    double s = sigmoid(x);
    return s * (1 - s);
}

double activationFunction(double x)
{
    //use a fucntion with out range [ -1 to 1 ]
    // ill use tanh for now
    return tanh(x);
}
double activationFunctionDerivative(double x)
{
    // this approximates the derivitive of tanh. pros: faster. con: less accurate 
    return 1.0 - x*x;
}

class Neuron 
{
public:
    double value, delta;
    // keep track of which neurons are dropped in training 
    bool dropped;
    vector<double> weights;
    // constructor/init
    Neuron(size_t num_outputs); 
    //deconstructor
    ~Neuron(); 
};

Neuron::Neuron(size_t num_outputs) //constructor
{ 
    // initialize dropped as false
    dropped = false;  
    // push back a an output with a rand weight until num_outputs is reached 
    for (size_t i = 0; i < num_outputs; ++i) 
    {
        // add with rand weight
        weights.push_back(((double)rand() / RAND_MAX) * 2 - 1); 
    }
}
Neuron::~Neuron()//deconstructor
{ 
    // cout << "Neuron Dropped" << endl;

}

class Layer 
{
public:
    // a Layer is a vector of neurons
    vector<Neuron> neurons;

    double running_mean = 0.0;
    double running_variance = 1.0;

    //constructor/init
    Layer(size_t num_neurons, size_t num_outputs); 
    // deconstructor
    ~Layer(); 
};

Layer::Layer(size_t num_neurons, size_t num_outputs)  //constructor
{
    // to construct a Layer by pushsing back Neurons until num_neurons is reached 
    for (size_t i = 0; i < num_neurons; ++i) 
    {
        // add neurons with num_outputs number of outputs(with rand weight) to the layer
        neurons.push_back(Neuron(num_outputs));
    }
}
Layer::~Layer()  //deconstructor
{
    //cout << "Layer deleted" <<endl;
}

class MLP 
{
private:

    bool apply_batch;

    void setInput(const vector<double> &input);

    void calculateOutputLayerDeltas(const vector<double> &target);

    // private functions 

    void propagateForward(bool training);

    void propagateBackward();

    void updateWeights(double learning_rate);

    void applyDropout(bool training);

    void normalizeBatch(const size_t layer_index, bool training, double momentum = 0.9, double epsilon = 1e-5);

    void normalizeBatchBackward(const size_t layer_index, const double epsilon = 1e-5);

public:
    // public variables
    vector<Layer> layers;
    double dropout_rate;

    // Constructors 
    // constructor/init 
    MLP(const vector<size_t> &topology, double dropout_rate = 0.1, bool apply_batch = true); 

    // deconstructor 
    ~MLP();

    // Functions
    void forward(const vector<double> &input, bool training);

    void backward(const vector<double> &target, double learning_rate);

    const vector<double> get_output() const;
    
};


MLP::MLP(const vector<size_t> &topology, double dropout_rate, bool apply_batch) //constructor
{ 
    this->apply_batch = apply_batch;
    // set dropout_rate to the rate passed by the user 
    if (dropout_rate != 0.0)
    {
        cout << "Dropout : " << dropout_rate*100 << "%" << endl;
    }
    else{
        cout << "No Dropout" << endl;
    }
    this->dropout_rate = dropout_rate;
    
    // loop through the topology from start to 1 before the output layer
    for (size_t i = 0; i < topology.size() - 1; ++i) 
    {
        // add a layer with topology[i] neurons and topology[i + 1] output neurons 
        layers.push_back(Layer(topology[i], topology[i + 1]));
    }
    // push back the output layer (which has topology.back() input neurons and 0 output neurons)
    layers.push_back(Layer(topology.back(), 0)); 
}

MLP::~MLP()
{
    //cout << "Multi Layer Perceptron deleted" <<endl;
}

void MLP::calculateOutputLayerDeltas(const vector<double> &target) 
{
    // loop through neurons in output layer (layer.back())
    for (size_t i = 0; i < layers.back().neurons.size(); ++i) 
    {
        // calculate delta
        layers.back().neurons[i].delta = (target[i] - layers.back().neurons[i].value) * activationFunctionDerivative(layers.back().neurons[i].value);
    }
}

void MLP::forward(const vector<double> &input, bool training) 
{
    // input vals are passed to input neurons 
    setInput(input);
    // do forward propagation to get output of each neuron 
    propagateForward(training);
    
    // Apply batch normalization (except for the output layer)
    if (apply_batch) 
    {
        for (size_t i = 1; i < layers.size() - 1; ++i) // Exclude the input and output layers
        {
            normalizeBatch(i, training);
        }
    }

    // when training apply dropout (which drops random neurons to prevent overfitting)
    applyDropout(training);
}

void MLP::backward(const vector<double> &target, double learning_rate) 
{
    // calculate output layer error (aka delta) given the target vals
    calculateOutputLayerDeltas(target);
    // propagate backwards calculating delta along the way 
    propagateBackward();

    // Apply batch normalization backward pass (except for the output layer)
    if (apply_batch)
    {
        for (size_t i = layers.size() - 2; i > 0; --i) // Exclude the input and output layers
        {
            normalizeBatchBackward(i);
        }
    }
    // update the weights based on the calculated deltas and the learninging rate (alpha)
    updateWeights(learning_rate); 
}

// this function just reads neuron vals without changing vals, hince the const 
const vector<double> MLP::get_output() const 
{ 
    // initilize a vecotor of doubles (called output)
    vector<double> output;
    // loop through output neurons (layers.back())
    for (const Neuron &neuron : layers.back().neurons) 
    {
        // add values to output vector 
        output.push_back(neuron.value);
    }
    // return the output vector that contains the output neuron values 
    return output;
}

void MLP::setInput(const vector<double> &input) 
{
    // for weather we have 13 input features  
    // Dew Max, Dew Avg, Dew Min, Humidity Max, Humidity Avg, Humidity Min, Wind Speed Max, Wind Speed Avg, Wind Speed Min, Pressure Max, Pressure Avg, Pressure Min, Precipitation Total

    // loop through in vector and set the input neurons to the input vals
    for (size_t i = 0; i < input.size(); ++i) 
    {
        // input (layer 0) neurons values are set to the input vector values 
        // cout << "Input " << i << " : " << input[i] << endl;
        layers[0].neurons[i].value = input[i];
    }
}

void MLP::propagateForward(bool training)
{
    // loop through layers in the MLP (starting at the first hidden layer)
    for (size_t i = 1; i < layers.size(); ++i) 
    {
        // loop through neurons in the layer
        for (size_t j = 0; j < layers[i].neurons.size(); ++j) 
        {
            // going to get the weighted sum 
            double sum = 0;
            // loop through the neurons in prev layer
            for (size_t k = 0; k < layers[i - 1].neurons.size(); ++k) 
            {
                // add the sum of the neurons in prev layer (including the weights in connection to current neuron)
                sum += layers[i - 1].neurons[k].value * layers[i - 1].neurons[k].weights[j];
            }
            // send the sum to the transfer function and set the result to the neurons output val
            layers[i].neurons[j].value = activationFunction(sum);
        }
        
    }
}


void MLP::propagateBackward()
{
    // loop through layers in the MLP in Reverse Order
    for (size_t i = layers.size() - 2; i > 0; --i)
    {
        // loop through neurons in the layer
        for (size_t j = 0; j < layers[i].neurons.size(); ++j)
        {
            // going to get the error
            double error = 0;
            // loop through the neurons in next layer
            for (size_t k = 0; k < layers[i + 1].neurons.size(); ++k)
            {
                // add the error of the neurons in next layer (including the delta in connection to next neuron)
                error += layers[i].neurons[j].weights[k] * layers[i + 1].neurons[k].delta;
            }
            // calculate delta by multiplying the error and the derivative of the transfer (activation) function
            layers[i].neurons[j].delta = error * activationFunctionDerivative(layers[i].neurons[j].value);
        }
    }
}

void MLP::updateWeights(double learning_rate) 
{
    // loop through layers in the MLP
    for (size_t i = 0; i < layers.size() - 1; ++i) 
    {
        // loop through neurons in the layer
        for (size_t j = 0; j < layers[i].neurons.size(); ++j) 
        {
            // loop through weights in the neuron
            for (size_t k = 0; k < layers[i].neurons[j].weights.size(); ++k) 
            {
                // update weight via multipluing learning rate (alpha) with the current val amd the delta of next neuron
                layers[i].neurons[j].weights[k] += learning_rate * layers[i].neurons[j].value * layers[i + 1].neurons[k].delta;
            }
        }
    }
}

void MLP::applyDropout(bool training) 
{
    // cout << "dropout_rate " << dropout_rate <<endl;
    // skip first and last layers
    for (size_t i = 1; i < layers.size() - 1; ++i) 
    { 
        // loop through all nuerons 
        // for (auto &neuron : layers[i].neurons) 
        for (size_t j = 0; j < layers[i].neurons.size(); ++j)
        {
            // if in the training phase 
            if (training == true) 
            {
                //get rand num between 1 and 0
                double dropout = ((double)rand() / RAND_MAX);
                //if that rand num is below dropout_rate then set neuron to 0
                if (dropout < dropout_rate) 
                {
                    //this effectivly "drops" the neuron since it no longer contributes to network
                    layers[i].neurons[j].value = 0;
                    // set drop to true 
                    layers[i].neurons[j].dropped = true;
                }  
                else
                {
                    layers[i].neurons[j].dropped = false;
                }

            } 
            else 
            {
                layers[i].neurons[j].dropped = false; // this turns them back white on eval
            }
        }
    }
}

void MLP::normalizeBatch(const size_t layer_index, bool training, double momentum, double epsilon)
{
    Layer &layer = layers[layer_index];
    double mean;
    double variance;

    if (training)
    {
        // calculate the mean and variance of the neuron values in the layer
        mean = 0.0;
        variance = 0.0;
        // using iterators to change it up a bit
        for (auto it = layer.neurons.begin(); it != layer.neurons.end(); ++it)
        {
            mean += it->value;
        }
        mean = mean / layer.neurons.size();

        for (auto it = layer.neurons.begin(); it != layer.neurons.end(); ++it)
        {
            variance += (it->value - mean) * (it->value - mean);
        }
        variance = variance / layer.neurons.size();

        // if training, update the running mean and variance here
        layer.running_mean = momentum * layer.running_mean + (1.0 - momentum) * mean;
        layer.running_variance = momentum * layer.running_variance + (1.0 - momentum) * variance;
    }
    else
    {
        // if testing, update the running mean and variance for normalization here
        mean = layer.running_mean;
        variance = layer.running_variance;
    }

    // normalize the neuron values based on the mean and variance
    for (auto it = layer.neurons.begin(); it != layer.neurons.end(); ++it)
    {
        it->value = (it->value - mean) / sqrt(variance + epsilon);
    }
}

void MLP::normalizeBatchBackward(const size_t layer_index, const double epsilon)
{
    Layer &layer = layers[layer_index];

    double mean = 0;
    double variance = 0;
    // using iterators to change it up a bit
    for (auto it = layer.neurons.begin(); it != layer.neurons.end(); ++it)
    {
        mean += it->value;
    }
    mean = mean/ layer.neurons.size();

    for (auto it = layer.neurons.begin(); it != layer.neurons.end(); ++it)
    {
        variance += (it->value - mean) * (it->value - mean);
    }
    variance = variance/layer.neurons.size();

    // find standard devaition 
    double std_dev = sqrt(variance + epsilon);
    // find inverse of it
    double inv_std_dev = 1.0 / std_dev;

    // calculate derivative of the loss with respect to x for each neuron 
    for (auto it = layer.neurons.begin(); it != layer.neurons.end(); ++it)
    {
        // multiply delta by the inverse of the standard dev
        it->delta = it->delta * inv_std_dev;
    }
}


// **** prototype functions (implemented below main) ****

// parsing fucntion takes in a csv and two empty matricies, fills them with the normalization vals and true vals and returns a matrix of the csv
Matrix parseCSV(const string& filename, Matrix& norm_vals, Matrix& true_vals);
// split function splits the matrix at a collumn and saves it to left_mtx and right_mtx 
void splitMatrixAtColumn(const Matrix& inputMatrix, int split_col_idx, Matrix& left_mtx, Matrix& right_mtx);
// graphviz creates a dot file of the MLP and topology that is passed
void graphviz(const string& filename,const MLP& mlp, const vector<size_t> topology, bool display_layer_stats = false);

int main() 
{
    srand(time(0));

    // vector<size_t> topology = {2, 3, 1}; //input {2 input neurons, 1 hidden layer (with 3 neurons), 1 p}
    unsigned int num_outputs = 3; // unsigned since it will always be positive 
    // vector<size_t> topology = {3, 64, 32, 16, num_outputs};
    vector<size_t> topology = {13, 64, 32, 16, num_outputs}; //13 input features, 3 outputs
    // vector<size_t> topology = {5, 10, 12, 10, num_outputs}; //13 input features, 3 outputs
    // MLP mlp(topology, .4);

    double dropout_rate = 0.0;
    
    size_t num_epochs = 2;

    double learning_rate = 0.01; // This is my alpha

    bool apply_batch = true;

    cout << endl;
    
    MLP mlp(topology, dropout_rate, apply_batch);
    string extention;
    if (apply_batch)
    {
        extention = "bn";
        cout << "Batch Normalization Applied";
    }
    else
    {
        extention = "sk";
        cout << "Batch Normalization Skipped";
    }

    // ***************** TRAINING ***************************** 
    string train_file_name = "Month_Data/April_Data.csv";
    Matrix train_norm_vals; //init here and pass by refrence 
    Matrix train_true_vals;
    Matrix train_mtx = parseCSV(train_file_name, train_norm_vals, train_true_vals); // this parses and normalizes the matrix

    // train_mtx.display();

    int split_idx = num_outputs;
    Matrix left_mtx, right_mtx;
    splitMatrixAtColumn(train_mtx, split_idx, left_mtx, right_mtx);

    

    Matrix train_inputs = right_mtx;
    Matrix targets = left_mtx;

    // train_inputs.display();

    // cout << "True : " << endl;
    // true_vals.display();
    // cout << endl;

    
    
    // training loop 
    for (size_t epoch = 0; epoch < num_epochs; ++epoch) 
    {
        // iterate through the samples 
        for (size_t i = 0; i < train_inputs.rows(); ++i) 
        {
            // forward calles the setInput function
            // then perform forward propagation 
            mlp.forward(train_inputs[i], true); // passing 'true' since this is the training step

            // perform backwards popagation to update weights
            mlp.backward(targets[i], learning_rate);
        }
    }

    graphviz("train_"+ extention + ".dot", mlp, topology, true); // mlp and topology are passed by reference 

    // ***************** TESTING ***************************** 
    string test_file_name = "Month_Data/March_2023.csv";
    // string test_file_name = "Month_Data/April_2023.csv";
    Matrix test_norm_vals; //init here and pass by refrence 
    Matrix test_true_vals;
    Matrix test_mtx = parseCSV(test_file_name, test_norm_vals, test_true_vals); // this parses and normalizes the matrix

    // test_mtx.display();
    // test_norm_vals.display();
    // test_true_vals.display();

    // int split_idx = num_outputs;
    Matrix test_left_mtx, test_right_mtx;
    splitMatrixAtColumn(test_mtx, split_idx, test_left_mtx, test_right_mtx);

    cout << endl;

    Matrix test_inputs = test_right_mtx;
    // test_inputs.display();
    // Matrix test_targets = test_left_mtx; // Normalized test values


    // Test the trained MLP
    double tmax_min_val  = test_norm_vals[0][0];
    double tmax_max_val  = test_norm_vals[0][1];

    double tavg_min_val  = test_norm_vals[1][0];
    double tavg_max_val  = test_norm_vals[1][1];

    double tmin_min_val  = test_norm_vals[2][0];
    double tmin_max_val  = test_norm_vals[2][1];

    
    // cout << tmax_min_val << " " << tmax_max_val<< endl;
    // cout << tavg_min_val << " " << tavg_max_val << endl;
    // cout << tmin_min_val << " " << tmin_max_val << endl;

    // set to two decimals 
    cout << fixed << setprecision(2);

    long day = 1;

    VectorOps tmax_delta;
    VectorOps tavg_delta;
    VectorOps tmin_delta;

    double tmax_squared_sum = 0.0;

    double tavg_squared_sum = 0.0;

    double tmin_squared_sum = 0.0;

    for (size_t i = 0; i < test_inputs.rows(); ++i) 
    {
        // Access the i-th row in inputs 
        vector<double>& input = test_inputs[i];
        // passing 'false' since this is the Evaluation step
        mlp.forward(input, false); 

        double out_0 =  mlp.get_output()[0] * (tmax_max_val - tmax_min_val) + tmax_min_val; // val * range + min
        double out_1 =  mlp.get_output()[1] * (tavg_max_val - tavg_min_val) + tavg_min_val; // val * range + min
        double out_2 =  mlp.get_output()[2] * (tmin_max_val - tmin_min_val) + tmin_min_val; // val * range + min

        double real_out_0 = test_true_vals[day-1][0];
        double real_out_1 = test_true_vals[day-1][1];
        double real_out_2 = test_true_vals[day-1][2];

        double diff_0 = abs(out_0 - real_out_0);
        tmax_delta.add(diff_0);
        tmax_squared_sum += diff_0 * diff_0;

        double diff_1 = abs(out_1 - real_out_1);
        tavg_delta.add(diff_1);
        tavg_squared_sum += diff_1 * diff_1;

        double diff_2 = abs(out_2 - real_out_2);
        tmin_delta.add(diff_2);
        tmin_squared_sum += diff_2 * diff_2;
        // cout << "Input: (" << input[0] << ", " << input[1] << ", " << input[2] << ", " << input[3] << ", " << input[4 ]<< ", ... ) ";

        cout << day << ": ";
        cout << out_0 <<" - "<< real_out_0 << ":  " << diff_0 << ",   ";
        cout << out_1 <<" - "<< real_out_1 << ":  " << diff_1 << ",   " ;
        cout << out_2 <<" - "<< real_out_2 << ":  " << diff_2 << endl;
        day++;
    }

    graphviz("test_"+ extention + ".dot", mlp, topology); // mlp and topology are passed by reference
    cout << endl;
    // Temp Max, delta between pred and real
    cout << "Temp Max" << endl;
    // tmax_delta.display_vect();
    // calculate the stats after all data has been added for better performance
    tmax_delta.calc_stats();
    //  Display performance data
    // tmax_delta.display_stats();
    // cout << endl;

    // Display Mean Absolute Error
    cout << "  Temp Max MAE: " << tmax_delta.mean << endl;
    // Display Root Mean Squared Error
    cout << "  Temp Max RMSE: " << sqrt(tmax_squared_sum / (day-1)) << endl;
    
    cout << endl;

    // Temp Avg, delta between pred and real
    cout << "Temp Avg Delta" << endl;
    // tavg_delta.display_vect();
    // calculate the stats after all data has been added for better performance
    tavg_delta.calc_stats();
    // display calculated stats 
    // tavg_delta.display_stats();
    // cout << endl;

    // Display Mean Absolute Error
    cout << "  Temp Avg MAE: " << tavg_delta.mean << endl;
    // Display Root Mean Squared Error
    cout << "  Temp Avg RMSE: " << sqrt(tavg_squared_sum / (day-1)) << endl;

    cout << endl;

    // Temp Min, delta between pred and real
    cout << "Temp Min Delta" << endl;
    // tmin_delta.display_vect();
    // calculate the stats after all data has been added for better performance
    tmin_delta.calc_stats();
    // display calculated stats 
    // tmin_delta.display_stats();
    // cout << endl;

    // Display Mean Absolute Error
    cout << "  Temp Min MAE: " << tmin_delta.mean << endl;
    // Display Root Mean Squared Error
    cout << "  Temp Min RMSE: " << sqrt(tmin_squared_sum / (day-1)) << endl;
    cout << endl;

    return 0;
}

Matrix parseCSV(const string& filename, Matrix& norm_vals, Matrix& true_vals) 
{
    ifstream input(filename);
    vector<vector<string>> data;
    string line;

    long i = 0;
    while (getline(input, line)) 
    {
        ++i;
        if (i <= 2) //skip first two rows
        { 
            continue;
        }
        stringstream ss(line);
        vector<string> row;

        string value;
        getline(ss, value, ','); // grab the date so that its not in the row data
        while (getline(ss, value, ',')) 
        {
            row.push_back(value);
        }
        data.push_back(row);
        
    }

    Matrix matrix;

    for (int i = 0; i < data.size(); ++i) 
    {
        vector<double> row;
        vector<double> true_row;
        for (int j = 0; j < data[i].size(); ++j) 
        {
            double temp = stof(data[i][j]); //string to float
            row.push_back(temp); 

            // Store the first three columns in true_row
            if (j < 3)  
            { 
                true_row.push_back(temp);
            }
        }
        matrix.push_back(row);
        true_vals.push_back(true_row);
    }

    // Matrix norm_vals;
    for (int j = 0; j < matrix.cols(); ++j) //Normalize the data
    { 
        vector<double> norm_row;
        double min_val = matrix[0][j];
        double max_val = matrix[0][j];
        
        for (int i = 0; i < matrix.rows(); ++i) 
        {
            if (matrix[i][j] < min_val) 
            {
                min_val = matrix[i][j];
            }
            if (matrix[i][j] > max_val) 
            {
                max_val = matrix[i][j];
            }
        }
        double range = max_val - min_val;

        norm_row.push_back(min_val);
        norm_row.push_back(max_val);
        norm_vals.push_back(norm_row);

        for (int i = 0; i < matrix.rows(); ++i) 
        {
            matrix[i][j] = (matrix[i][j] - min_val) / range;
        }
    }
    
    return matrix;
}

void splitMatrixAtColumn(const Matrix& inputMatrix, int split_col_idx, Matrix& left_mtx, Matrix& right_mtx) 
{
    int rows = inputMatrix.rows();
    int cols = inputMatrix.cols();

    left_mtx.resize(rows, split_col_idx);
    right_mtx.resize(rows, cols - split_col_idx);

    for (int i = 0; i < rows; i++) 
    {
        for (int j = 0; j < split_col_idx; j++) 
        {
            left_mtx[i][j] = inputMatrix[i][j];
        }
        for (int j = split_col_idx; j < cols; j++) 
        {
            right_mtx[i][j - split_col_idx] = inputMatrix[i][j];
        }
    }

}

// Create Graphviz file
void graphviz(const string& filename,const MLP& mlp, const vector<size_t> topology, bool display_layer_stats) 
{
    ofstream file;
    file.open(filename);
    file << "digraph G {\n";
    file << "  rankdir=LR;\n";
    
    // add the nodes
    for (size_t i = 0; i < topology.size(); ++i) 
    {
        VectorOps temp_vector;
        file << "  subgraph cluster_" << i << " {\n";
        file << "    style=filled;\n";
        file << "    color=lightgrey;\n";
        file << "    node [style=filled,color=white];\n";
        for (size_t j = 0; j < topology[i]; ++j) 
        {
            string node_color = mlp.layers[i].neurons[j].dropped ? "red" : "white"; // if dropped set to red
            string neuron_val = to_string(mlp.layers[i].neurons[j].value); // get neuron value
            neuron_val.resize(4); // truncate after 2 decimal places (aka 4 chars)
            file << "    i" << i << "h" << j << " [style=filled, color=" << node_color << ", label=\"" << neuron_val << "\"];\n";
            temp_vector.add(mlp.layers[i].neurons[j].value);
        }
        file << "    label = \"Layer " << i << "\";\n";
        file << "  }\n";
        if (display_layer_stats)
        {
            cout <<endl;
            cout << "Layer " << i  <<endl;
            temp_vector.calc_stats();
            temp_vector.display_stats();
            
        }
        
    }
    
    // add the arrows 
    for (size_t i = 0; i < topology.size() - 1; ++i) 
    {
        for (size_t j = 0; j < topology[i]; ++j) 
        {
            for (size_t k = 0; k < topology[i + 1]; ++k) 
            {
                // Add the connection between nodes without specifying color or label
                file << "  i" << i << "h" << j << " -> i" << (i + 1) << "h" << k << ";\n";
            }
        }
    }

    file << "}\n";
    file.close();

    
}