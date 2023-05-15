#include <iostream>
#include <thread>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <chrono>
#include <thread>
namespace py = pybind11;

using std::cout;
using std::cin;
using std::vector;
using std::string;
using std::max;
using std::min;
using std::invalid_argument;
using std::endl;


py::array_t<double> matmul(vector<vector<double>> A, vector<vector<double>> B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw invalid_argument("The number of columns in matrix A must be equal to the number of rows in matrix B.");
    }

    vector<vector<double>> C(rowsA, vector<double>(colsB));

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    size_t N = C.size();
    size_t M = C[0].size();

    py::array_t<double, py::array::c_style> arr({N, M});

    auto ra = arr.mutable_unchecked();

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            ra(i, j) = C[i][j];
        };
    };

    return arr;
}


py::array_t<double> conv2d_single_channel(vector<vector<double>> Input, vector<vector<double>> kernel, string padding = "valid", int strides = 1, int pad_size = 1) {
    bool multichannel = false;
    int xKernShape = kernel.size();
    int yKernShape = kernel[0].size();
    int xInput = Input.size();
    int yInput = Input[0].size();
    int nChannels = 1;
    int nfilters = 1;

    vector<vector<double>> output;
    vector<vector<double>> InpPadded;
    if (padding == "valid") {
        pad_size = 0;
        InpPadded = Input;
    }
    else if (padding == "same" && strides == 1) {
        pad_size = 1;
        InpPadded.resize(xInput + 2 * pad_size, vector<double>(yInput + 2 * pad_size));
        for (int i = pad_size; i < xInput + pad_size; i++) {
            for (int j = pad_size; j < yInput + pad_size; j++) {
                InpPadded[i][j] = Input[i - pad_size][j - pad_size];
            }
        }
    }
    else if (padding == "custom") {
        if (pad_size < 1) {
            cout << "ERROR : pad_size must be greater than equal to 1 for type-custom" << endl;
        }
        InpPadded.resize(xInput + 2 * pad_size, vector<double>(yInput + 2 * pad_size));
        for (int i = pad_size; i < xInput + pad_size; i++) {
            for (int j = pad_size; j < yInput + pad_size; j++) {
                InpPadded[i][j] = Input[i - pad_size][j - pad_size];
            }
        }
    }

    int xOutput = ((xInput - xKernShape + 2 * pad_size) / strides) + 1;
    int yOutput = ((yInput - yKernShape + 2 * pad_size) / strides) + 1;

    output.resize(xOutput, vector<double>(yOutput));
    int j = 0;

    for (int y = 0; y < InpPadded[0].size(); y++) {
        if (y > InpPadded[0].size() - yKernShape) {
            break;
        }
        if (y % strides == 0) {
            int i = 0;
            for (int x = 0; x < InpPadded.size(); x++) {
                if (x > InpPadded.size() - xKernShape) {
                    break;
                }
                if (x % strides == 0) {
                    double patch_sum = 0.0;
                    for (int m = x; m < x + xKernShape; m++) {
                        for (int n = y; n < y + yKernShape; n++) {
                            patch_sum += kernel[m - x][n - y] * InpPadded[m][n];
                        }
                    }
                    output[i][j] += patch_sum;
                    i++;
                }
            }
            j++;
        }
    }

    size_t N = output.size();
    size_t M = output[0].size();

    py::array_t<double, py::array::c_style> arr({N, M});

    auto ra = arr.mutable_unchecked();

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            ra(i, j) = output[i][j];
        };
    };
    return arr;
}

py::array_t<double> conv2d_multi_channel(vector<vector<vector<double>>> Input, vector<vector<vector<double>>> kernel, string padding = "valid", int strides = 1, int pad_size = 1) {
    bool multichannel = true;
    int xKernShape = kernel.size();
    int yKernShape = kernel[0].size();
    int nfilters = kernel[0][0].size();
    int xInput = Input.size();
    int yInput = Input[0].size();
    int nChannels = Input[0][0].size();

    vector<vector<double>> output;
    vector<vector<vector<double>>> InpPadded;
    if (padding == "valid") {
        pad_size = 0;
        InpPadded = Input;
    }
    else if (padding == "same" && strides == 1) {
        pad_size = 1;
        InpPadded.resize(xInput + 2 * pad_size, vector<vector<double>>(yInput + 2 * pad_size, vector<double>(nChannels)));
        for (int i = pad_size; i < xInput + pad_size; i++) {
            for (int j = pad_size; j < yInput + pad_size; j++) {
                for (int k = 0; k < nChannels; k++) {
                    InpPadded[i][j][k] = Input[i - pad_size][j - pad_size][k];
                }
            }
        }
    }
    else if (padding == "custom") {
        if (pad_size < 1) {
            cout << "ERROR : pad_size must be greater than equal to 1 for type-custom" << endl;
        }
        InpPadded.resize(xInput + 2 * pad_size, vector<vector<double>>(yInput + 2 * pad_size, vector<double>(nChannels)));
        for (int i = pad_size; i < xInput + pad_size; i++) {
            for (int j = pad_size; j < yInput + pad_size; j++) {
                for (int k = 0; k < nChannels; k++) {
                    InpPadded[i][j][k] = Input[i - pad_size][j - pad_size][k];
                }
            }
        }
    }

    int xOutput = ((xInput - xKernShape + 2 * pad_size) / strides) + 1;
    int yOutput = ((yInput - yKernShape + 2 * pad_size) / strides) + 1;

    output.resize(xOutput, vector<double>(yOutput));
    int j = 0;

    for (int y = 0; y < InpPadded[0].size(); y++) {
        if (y > InpPadded[0].size() - yKernShape) {
            break;
        }
        if (y % strides == 0) {
            int i = 0;
            for (int x = 0; x < InpPadded.size(); x++) {
                if (x > InpPadded.size() - xKernShape) {
                    break;
                }
                if (x % strides == 0) {
                    double patch_sum = 0.0;
                    for (int m = x; m < x + xKernShape; m++) {
                        for (int n = y; n < y + yKernShape; n++) {
                            for (int k = 0; k < nfilters; k++) {
                                patch_sum += kernel[m - x][n - y][k] * InpPadded[m][n][k];
                            }
                        }
                    }
                    output[i][j] += patch_sum;
                    i++;
                }
            }
            j++;
        }
    }

    size_t N = output.size();
    size_t M = output[0].size();

    py::array_t<double, py::array::c_style> arr({N, M});

    auto ra = arr.mutable_unchecked();

    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < M; j++)
        {
            ra(i, j) = output[i][j];
        };
    };

    return arr;
}

double vector_max(vector<vector<double>> inp){
    double max=inp[0][0];
    for(int i=0;i<inp.size();i++){
        for(int j=0;j<inp[0].size();j++){
            if(inp[i][j]>max){
                max=inp[i][j];
            }
        }
    }
    return max;
}

double vector_min(vector<vector<double>> inp){
    double min=inp[0][0];
    for(int i=0;i<inp.size();i++){
        for(int j=0;j<inp[0].size();j++){
            if(inp[i][j]<min){
                min=inp[i][j];
            }
        }
    }
    return min;
}

double vector_mean(vector<vector<double>> inp){
    double sum=0;
    for(int i=0;i<inp.size();i++){
        for(int j=0;j<inp[0].size();j++){
            sum+=inp[i][j];
        }
    }
    return sum/(inp.size()*inp[0].size());
}

double get_pool(vector<vector<double>> inp,const string ptype="max"){
    if(ptype=="max"){
        return(vector_max(inp));
    }else if(ptype=="min"){
        return(vector_min(inp));
    }else if(ptype=="mean"){
        return(vector_mean(inp));
    }
    return -1;
}

py::array_t<double>pool2d(vector<vector<double>> Input,const string pool_type="max",vector<int> size={2, 2},const string padding="valid",int strides=1,int pad_size=1){
    int xInput=Input.size();
    int yInput=Input[0].size();
    vector<vector<double>> output;
    vector<vector<double>> InpPadded;
    
    if(padding=="valid"){
        pad_size=0;
        InpPadded=Input;
    }else if(padding=="same" && strides==1){
        pad_size=1;
        InpPadded.resize(xInput + 2 * pad_size,vector<double>(yInput + 2 * pad_size));
        for (int i = pad_size; i < xInput + pad_size; i++) {
            for (int j = pad_size; j < yInput + pad_size; j++) {
                InpPadded[i][j] = Input[i - pad_size][j - pad_size];
            }
        }
    }else if(padding=="custom"){
        if(pad_size<1){
            cout<<"ERROR : pad_size must be greater than equal to 1 for type-custom"<<endl;
        }
        InpPadded.resize(xInput + 2 * pad_size,vector<double>(yInput + 2 * pad_size));
        for (int i = pad_size; i < xInput + pad_size; i++) {
            for (int j = pad_size; j < yInput + pad_size; j++) {
                InpPadded[i][j] = Input[i - pad_size][j - pad_size];
            }
        }
    }

    int xOutput=int(((xInput - size[0] + 2 * pad_size) / strides) + 1);
    int yOutput=int(((yInput - size[1] + 2 * pad_size) / strides) + 1);
    output.resize(xOutput,vector<double>(yOutput));

    if(pool_type=="global_max"){
        double data=vector_max(Input);
        py::array_t<double> vmax({1}, &data);
        return vmax(Input);
    }else if(pool_type=="global_min"){
        double data=vector_min(Input);
        py::array_t<double> vmin({1}, &data);
        return vmin;
    }else if(pool_type=="global_mean"){
        double data=vector_mean(Input);
        py::array_t<double> vmean({1}, &data);
        return vmean;
    }else{
        int j=0;
        for(int y=0;y<InpPadded[0].size();y++){
           if (y > InpPadded[0].size() - size[1]) {
            break;
        }
        if (y % strides == 0) {
            int i = 0;
            for (int x = 0; x < InpPadded.size(); x++) {
                if (x > InpPadded.size() - size[0]) {
                    break;
                }
                if (x % strides == 0) {
                    vector<vector<double>> resized_inp;
                    resized_inp.resize(size[0],vector<double>(size[1]));
                    for(int m=x,p=0;m<x+size[0];m++,p++) {
                        for(int n=y,q=0;n<y+size[1];n++,q++){
                            resized_inp[p][q]=InpPadded[m][n];
                        } 
                    }
                    output[i][j] += get_pool(resized_inp,pool_type);
                    i++;
                }
            }
            j++;
        } 
        }
    }

    size_t N = output.size();
    size_t M = output[0].size();

    py::array_t<double, py::array::c_style> arr({N, M});

    auto ra = arr.mutable_unchecked();

    for (size_t i = 0; i < N; i++){
        for (size_t j = 0; j < M; j++)
        {
            ra(i, j) = output[i][j];
        };
    };

    return arr;

}

PYBIND11_MODULE(neural_cpp_link, neural_link) {
  neural_link.def("py_matmul", &matmul,py::arg("A"),py::arg("B"));
  neural_link.def("py_conv2d_single_channel", &conv2d_single_channel,py::arg("Input"),py::arg("kernel"),py::arg("padding")="valid",py::arg("strides")=1,py::arg("pad_size")=1);
  neural_link.def("py_conv2d_multi_channel", &conv2d_multi_channel,py::arg("Input"),py::arg("kernel"),py::arg("padding")="valid",py::arg("strides")=1,py::arg("pad_size")=1);
  neural_link.def("py_get_pool", &get_pool,py::arg("inp"),py::arg("ptype")="max");
  neural_link.def("py_pool2d", &pool2d,py::arg("Input"),py::arg("pool_type")="max",py::arg("size")=vector<int>{2, 2},py::arg("padding")="valid",py::arg("strides")=1,py::arg("pad_size")=1);
}


