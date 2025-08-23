#include <cassert>
#include <cstdlib>
#include <print>
#define EIGEN_DEFAULT_IO_FORMAT Eigen::IOFormat(Eigen::FullPrecision, 0, ", ", ";\n", "", "", "[", "]") // Format like MATLAB
#include <Eigen/Core>
#include "to_string.hpp"

int main(int argc, char *argv[])
{
    std::println("Eigen version: {}.{}.{}", 
                 EIGEN_WORLD_VERSION, 
                 EIGEN_MAJOR_VERSION, 
                 EIGEN_MINOR_VERSION);

    std::println("Create a column vector:");
    Eigen::VectorXd x;
    // TODO
    x = (Eigen::Vector3d() << 1, 3.2, 0.01).finished();
    std::println("x = \n{}\n", to_string(x));

    std::println("Create a matrix:");
    Eigen::MatrixXd A(4, 3);
    // TODO: Don't just use a for loop or hardcode all the elements
    //       Try and be creative :)
    // create a vector of 4 elements
    Eigen::Vector4d p = (Eigen::Vector4d() << 1, 2, 3, 4).finished();
    // create a vector of 3 elements
    Eigen::Vector3d q = (Eigen::Vector3d() << 1, 2, 3).finished();
    // combine the two vectors into a matrix
    A = p * q.transpose();
    std::println("A.size() = {}", A.size());
    std::println("A.rows() = {}", A.rows());
    std::println("A.cols() = {}", A.cols());
    std::println("A = \n{}\n", to_string(A));
    std::println("A.transpose() = \n{}\n", to_string(A.transpose()));

    std::println("Matrix multiplication:");
    Eigen::VectorXd Ax;
    // TODO
    // multiply the matrix A by the vector x
    Ax = A * x;
    std::println("A*x = \n{}\n", to_string(Ax));

    std::println("Matrix concatenation:");
    // create a 4x6 matrix B 
    Eigen::MatrixXd B(4, 6);
    // TODO
    // concatenate the matrix A with itself
    B << A, A;
    std::println("B = \n{}\n", to_string(B));
    Eigen::MatrixXd C(8, 3);
    // TODO
    // concatenate the matrix A with itself
    C << A, A;
    std::println("C = \n{}\n", to_string(C));

    std::println("Submatrix via block:");
    Eigen::MatrixXd D(1, 3);
    // TODO
    // create a submatrix D that extracts from B using block operator
    D = B.block(1, 2, 1, 3);
    std::println("D = \n{}\n", to_string(D));
    std::println("Submatrix via slicing:");
    // TODO
    // create a submatrix D that extracts from B using slicing 
    D = B(Eigen::seq(1, 1), Eigen::seq(2, 4));
    std::println("D = \n{}\n", to_string(D));

    std::println("Broadcasting:");
    Eigen::VectorXd v;
    Eigen::MatrixXd E;
    v = (Eigen::VectorXd(6) << 1, 3, 5, 7, 4, 6).finished();
    // TODO
    // add B and v together using broadcasting
    E = B.rowwise() + v.transpose();
    std::println("E = \n{}\n", to_string(E));

    std::println("Index subscripting:");
    Eigen::MatrixXd F(4,6);
    // TODO
    // create two arrays of indices
    Eigen::VectorXi r(4);
    Eigen::VectorXi c(6);
    r << 0, 2, 1, 3;
    c << 0, 3, 1, 4, 2, 5;
    F = B(r, c);
    std::println("F = \n{}\n", to_string(F));

    std::println("Memory mapping:");
    float array[9] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> G(array);              // TODO: Replace this with an Eigen::Map
    array[2] = -3.0f;               // Change an element in the raw storage
    assert(array[2] == G(0,2));     // Ensure the change is reflected in the view
    G(2,0) = -7.0f;                 // Change an element via the view
    assert(G(2,0) == array[6]);     // Ensure the change is reflected in the raw storage
    std::println("G = \n{}\n", to_string(G));

    return EXIT_SUCCESS;
}
