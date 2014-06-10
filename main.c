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
#include <unistd.h>

#define DEFAULT_ROW 10
#define DEFAULT_COL 4
#define DEFAULT_H 0.1
#define DEFAULT_DT 0.001

#define MIN 0.0
#define MAX 1.0
#define TAG 123
#define TAG2 456
#define MAX_ITER 500

double h, dt, pow_h, elapsed, start, end;
int ROW, COL, size;

double initializeValue(int row, int col, int N, int M, int my_rank, int size) {
    if (my_rank == 0 && (col == 0 || (row == N - 1 || row == 0))) {
        return 0.0;
    } else if (my_rank == size - 1 && (col == M - 1 || (row == N - 1 || row == 0))) {
        return 0.0;
    } else if (my_rank < size - 1 && my_rank > 0 && (row == N - 1 || row == 0)) {
        return 0.0;
    } else {
        return 1.0;
    }
}

double** initializeTable(double** tab, int my_rank, int size) {

    tab = calloc(COL, sizeof (double*));

    int row, col;
    for (col = 0; col < COL; col++) {
        tab[col] = calloc(ROW, sizeof (double));
    }

    for (row = 0; row < ROW; row++) {
        for (col = 0; col < COL; col++) {

            tab[col][row] = initializeValue(row, col, ROW, COL, my_rank, size);
        }
    }
    return tab;
}

void printfTable(double** tab, int rank) {
    printf("\ntab%d start:\n", rank);
    int row, col;

    for (col = 0; col < COL; col++) {
        for (row = 0; row < ROW; row++) {
            printf("%1.3f |", tab[col][row]);
        }
        printf("\n");
    }
    printf("tab%d end\n", rank);
}

void printfVector(double* vector, int rank) {
    printf("\nvector%d start:\n", rank);
    int row;
    for (row = 0; row < ROW; row++) {
        printf("%1.3f\n", vector[row]);
    }
    printf("vector%d end\n", rank);
}

void saveTable(double **table, int rank, int i, int clear) {
    char fileName[40];
    FILE *file;
    sprintf(fileName, "result%d.txt", rank);

    if (clear == 1) {
        file = fopen(fileName, "wb");
    } else {
        file = fopen(fileName, "a");
    }
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    fprintf(file, "\n====================================== %d ======================================\n", i);
    int row, col;



    for (col = 0; col < COL; col++) {
        for (row = 0; row < ROW; row++) {
            fprintf(file, "%1.3f | ", table[col][row]);
        }
        fprintf(file, "\n\n");
    }
    fclose(file);
}

void saveVectors(double *topVector, double *bottomVector, int rank) {
    char fileName[40];
    FILE *file;
    sprintf(fileName, "result%d.txt", rank);

    file = fopen(fileName, "a");

    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    int row;
    fprintf(file, "\ntopVector:\n");
    for (row = 0; row < ROW; row++) {
        fprintf(file, "%1.3f | ", topVector[row]);
    }
    fprintf(file, "\n\nbottomVector:\n");
    for (row = 0; row < ROW; row++) {
        fprintf(file, "%1.3f | ", bottomVector[row]);
    }
    fprintf(file, "\n");
    fclose(file);
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

    double** copy = calloc(COL, sizeof (double*));
    int row, col;

    for (col = 0; col < COL; col++) {
        copy[col] = calloc(ROW, sizeof (double));
    }

    for (col = 0; col < COL; col++) {
        for (row = 0; row < ROW; row++) {
            copy[col][row] = tab[col][row];
        }
    }
    return copy;
}

double* getVector(double** tab, int col) {
    int row;
    double *vector;
    //    vector = calloc(ROW, sizeof (double));
    //    printf("row %d\n", row);
    //    printf("vector size %d\n", sizeof (vector) / sizeof (double) * N);
    //    printfTable(tab, 000);
    //    printf("M %d\n", M);
    //    printf("N %d\n", N);
    //    printf("tab[%d][%d]=%1.f\n", 1, 18, tab[1][18]);
    for (row = 0; row < ROW; row++) {
        vector[row] = tab[col][row];
        //                printf("tab[%d][%d]=%1.f\n", row, col, tab[row][col]);
        //        printf("vector[%d]=%1.f\n", row, vector[row]);
    }
    return vector;
}

//double* initializeVector(double *vector) {
////    double *vector;
//    int row;
//    for (row = 0; row < ROW; row++) {
//        vector[row] = 0;
//    }
//    return vector;
//}

void run(double** tab, int rank, MPI_Comm comm_cart) {

    double **prevTab = makeCopy(tab), bottomVector[COL], topVector[ROW];
    //    double topVector[ROW];
    //    initializeVector(topVector);
    //    double bottomVector[COL];
    //    initializeVector(bottomVector);
    int i, row, col, right, left, direction = 0, disp = 1;

    MPI_Status recv_status, recv_status2;
    MPI_Request request, request2;

    //    printf("prevTab size %d\n", sizeof (prevTab) / sizeof (double) * M * N);
    //    printf("tab size %d\n", sizeof (tab) / sizeof (double) * M * N);

    MPI_Cart_shift(comm_cart, direction, disp, &left, &right);

    //    if (rank == 0) {
    //    printf("rank %d left %d\n", rank, left);
    //    printf("rank %d right %d\n", rank, right);
    //    }
    //    double *aaa = calloc(COL, sizeof (double*));
    saveTable(tab, rank, 0, 1);
    for (i = 0; i < MAX_ITER; i++) {

        //        int sizeOf = sizeof (topVector) / sizeof (double) * ROW;
        //        topVector = getVector(tab, 0);
        //        bottomVector = getVector(tab, COL - 1);
        //        topVector= calloc(ROW, sizeof (double));
        //        bottomVector= calloc(ROW, sizeof (double));
        saveVectors(tab[0], tab[COL - 1], rank);
        //        printf("topVector size %d\n", sizeOf);
        //        printf("bottomVector size %d\n", sizeOf);
        //        break;
        //        double a = 0.1;
        //        double b, fromLeft[], fromRight[];
        //        if(rank==0) {
        //            printf("\n$$$$$%d\n", &topVector);
        //        }
        if (right >= 0 && right <= size) {
            //            printf("rank %d sent to right %d\n", rank, right);
            //            bottomVector = getVector(tab, COL - 1);
            MPI_Isend(tab[COL - 1], ROW, MPI_DOUBLE, right, TAG, comm_cart, &request);
            MPI_Recv(topVector, ROW, MPI_DOUBLE, right, TAG, comm_cart, &recv_status);
            //                        MPI_Wait(&request, &recv_status);
            //            printf("rank %d received from right\n", rank);
        }

        if (left >= 0 && left <= size) {
            //            printf("rank %d sent to left %d\n", rank, left);
            //            double c = 0.5, d;
            //            topVector = getVector(tab, 0);
            MPI_Isend(tab[0], ROW, MPI_DOUBLE, left, TAG, comm_cart, &request);

            MPI_Recv(bottomVector, ROW, MPI_DOUBLE, left, TAG, comm_cart, &recv_status);
            //            MPI_Wait(&request, &recv_status);
            //            printf("rank %d received from left\n", rank);
            //             printf("aa %d \n", fromRight[1]);
        }
        //        printf("\ntest\n");
        //        if (rank == 0) {
        //                    sleep(1);
        //            printfVector(bottomVector, rank);
        //        } else {
        //            printfVector(topVector, rank);
        //        }
        //                if(rank == 1) {

        //                    printf("\n$sizeof leftV %f\n", leftV);
        //                    printfVector(bottomVector, rank);
        //                }
        saveVectors(topVector, bottomVector, rank);
        //                saveVectors(leftV, rightV, rank);
        for (row = 0; row < ROW; row++) {
            for (col = 0; col < COL; col++) {
                double left = 0, right = 0, top = 0, bottom = 0;
                if (row > 0) {
                    top = prevTab[col][row - 1];

                }
                if (row < ROW - 1) {
                    bottom = prevTab[col][row + 1];
                }
                if (col < COL - 1) {
                    right = prevTab[col + 1][row];
                } else {
                    right = bottomVector[row];
                }
                if (col > 0) {
                    left = prevTab[col - 1][row];
                } else {
                    left = topVector[row];
                }
                tab[col][row] = calculateValue(left, right, top, bottom, prevTab[col][row]);
                prevTab[col][row] = tab[col][row];
            }
        }
        //        if (rank == 0) {
        //            double tabbb[ROW] = topVector;
        //            printf("\n## rank sent %d %p\n", rank, &tab[COL - 1]);
        //            printf("\n## topVector %p\n", &topVector);
        //            printf("\n## sent %p\n", &tab[0]);
        //            printf("\n## bottomVector %p\n", &bottomVector);
        //        }
        //        if (rank == 1) {
        //            double tabbb[ROW] = topVector;
        //            printf("\n## rank sent %d %p\n", rank, &tab[COL - 1]);
        //            printf("\n## topVector %p\n", &topVector);
        //            printf("\n## sent %p\n", &tab[0]);
        //            printf("\n## bottomVector %p\n", &bottomVector);
        //        }
        //        printf("i=%d\n", i + 1);
        //        break;
        //                printfTable(tab, rank);
        saveTable(tab, rank, i + 1, 0);

    }
    //    printfTable(tab, rank);

    //    printfVector(bottomVector, rank);

}

/*
 * Usage:
 *  ./pcam 100 100 0.001 0.01
 * 
 * Arguments initializes variables like below:
 * row = 100
 * col = 100
 * h = 0.001
 * dt = 0.01
 * 
 */
int main(int argc, char** argv) {

    if (argc < 7) {
        ROW = DEFAULT_ROW;
        COL = DEFAULT_COL;
        h = DEFAULT_H;
        dt = DEFAULT_DT;
    } else {
        ROW = atof(argv[2]);
        COL = atof(argv[1]);
        h = atof(argv[3]);
        dt = atof(argv[4]);
    }

    pow_h = h*h;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    start = MPI_Wtime();

    int ierror, my_rank, dims[2] = {size, 1}, periods[2] = {0, 0}, ndims = 2, reorder = 0;
    MPI_Comm comm_cart;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

    double** tab = initializeTable(tab, my_rank, size);

    //    printfTable(tab, my_rank);
    //    printf("rank %d\n", my_rank);

    run(tab, my_rank, comm_cart);

    //    printf("\nFinished!\n");
    MPI_Finalize();

    end = MPI_Wtime();
    elapsed = end - start;
    //    printf("\n\nRun time: %f\n", elapsed);

    return (EXIT_SUCCESS);
}



