/* 
 * File:   main.c
 * Author: Piotr Kozlowski
 *
 * PCAM implementation using MPI
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openmpi-x86_64/mpi.h>

#define DEFAULT_N 10
#define DEFAULT_M 10
#define DEFAULT_H 0.1
#define DEFAULT_DT 0.001

#define MIN 0.0
#define MAX 1.0
#define TAG 123
#define MAX_ITER 200

double h, dt, pow_h;
int N, M;

double initializeValue(int n, int m, int N, int M, int my_rank, int size) {
    if (my_rank == 0 && (m == 0 || (n == N - 1 || n == 0))) {
        return 0.0;
    } else if (my_rank == size - 1 && (m == M - 1 || (n == N - 1 || n == 0))) {
        return 0.0;
    } else if (my_rank < size - 1 && my_rank > 0 && (n == N - 1 || n == 0)) {
        return 0.0;
    } else {
        return 1.0;
    }
}

double** initializeTable(double** tab, int my_rank, int size) {

    tab = calloc(N, sizeof (double*));

    int n, m;
    for (n = 0; n < N; n++) {
        tab[n] = calloc(M, sizeof (double));
    }

    for (n = 0; n < N; n++) {
        for (m = 0; m < M; m++) {
            tab[n][m] = initializeValue(n, m, N, M, my_rank, size);
        }
    }
    return tab;
}

void printfTable(double** tab, int rank) {
    printf("\ntab%d start:\n", rank);
    int n, m;
    for (n = 0; n < N; n++) {
        for (m = 0; m < M; m++) {
            printf("%1.3f |", tab[n][m]);
        }
        printf("\n");
    }
    printf("tab%d end\n", rank);
}

double drand(double low, double high) {
    return ( (double) rand() * (high - low)) / (double) RAND_MAX + low;
}

double calculateValue(double left, double right, double top, double bottom, double previous) {
    double value = (dt * (((right + left + top + bottom) - (4 * previous)) / pow_h)) + previous;
    if (value > 0) {
        return value;
    } else {
        return 0;
    }
}

double** makeCopy(double** tab) {

    double** copy = calloc(N, sizeof (double*));
    int n, m;

    for (n = 0; n < N; n++) {
        copy[n] = calloc(M, sizeof (double));
    }

    for (n = 0; n < N; n++) {
        for (m = 0; m < M; m++) {
            copy[n][m] = tab[n][m];
        }
    }
    return copy;
}

double* getVector(double** tab, int col) {
    int m;
    double *vector;
    vector = calloc(M, sizeof (double));
    for (m = 0; m < M; m++) {
        vector[m] = tab[m][col];
    }
    return vector;
}

void run(double** tab, int rank, MPI_Comm comm_cart) {

    double** prevTab = makeCopy(tab), *leftVector, *rightVector;
    int i, n, m, right, left;

    MPI_Status recv_status;
    MPI_Request request;

    MPI_Cart_shift(comm_cart, 0, 1, &left, &right);

    for (i = 0; i < MAX_ITER; i++) {

        int sizeOf = sizeof (double) * M;
        leftVector = getVector(tab, 0);
        rightVector = getVector(tab, M);

        MPI_Isend(rightVector, sizeOf, MPI_DOUBLE, right, TAG, comm_cart, &request);
        MPI_Recv(&rightVector, sizeOf, MPI_DOUBLE, right, TAG, comm_cart, &recv_status);
        MPI_Wait(&request, &recv_status);

        MPI_Isend(leftVector, sizeOf, MPI_DOUBLE, left, TAG, comm_cart, &request);
        MPI_Recv(&leftVector, sizeOf, MPI_DOUBLE, left, TAG, comm_cart, &recv_status);
        MPI_Wait(&request, &recv_status);


        for (n = 0; n < N; n++) {
            for (m = 0; m < M; m++) {
                double left = 0, right = 0, top = 0, bottom = 0;
                if (n > 0) {
                    top = prevTab[n - 1][m];
                }
                if (n < N - 1) {
                    bottom = prevTab[n + 1][m];
                }
                if (m < M - 1) {
                    right = prevTab[n][m + 1];
                } else {
                    right = rightVector[m];
                }
                if (m > 0) {
                    left = prevTab[n][m - 1];
                } else {
                    left = leftVector[m];
                }
                tab[n][m] = calculateValue(left, right, top, bottom, prevTab[n][m]);
                prevTab[n][m] = tab[n][m];
            }

        }
        //        printf("i=%d\n", i);
//        break;
    }
    printfTable(tab, rank);
}

/*
 * Usage:
 *  ./pcam 100 100 0.001 0.01
 * 
 * Arguments initializes variables like below:
 * m = 100
 * n = 100
 * h = 0.001
 * dt = 0.01
 * 
 */
int main(int argc, char** argv) {

    if (argc < 7) {
        N = DEFAULT_N;
        M = DEFAULT_M;
        h = DEFAULT_H;
        dt = DEFAULT_DT;
    } else {
        N = atof(argv[2]);
        M = atof(argv[1]);
        h = atof(argv[3]);
        dt = atof(argv[4]);
    }

    pow_h = h*h;

    int size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int ierror, rank, my_rank, dims[2] = {1, size}, periods[2] = {0, 0}, ndims = 2, reorder = 0;
    MPI_Comm comm_cart;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

    double** tab = initializeTable(tab, my_rank, size);

    printfTable(tab, my_rank);

    run(tab, my_rank, comm_cart);

    printf("\nFinished!\n");
    MPI_Finalize();

    return (EXIT_SUCCESS);
}



