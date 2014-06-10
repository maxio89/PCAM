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
#define DEFAULT_COL 80
#define DEFAULT_H 0.1
#define DEFAULT_DT 0.001
#define DEFAULT_MAX_ITER 500

#define MIN 0.0
#define MAX 1.0
#define TAG 123

double h, dt, pow_h, elapsed, start, end;
int ROW, COL, size, maxIter;

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
    //    char *a;
    //    sprintf(a, "%d%d", col, row);
    //    return atof(a);
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

    for (row = 0; row < ROW; row++) {
        for (col = 0; col < COL; col++) {
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
    for (row = 0; row < ROW; row++) {
        for (col = 0; col < COL; col++) {

            fprintf(file, "%1.3f | ", table[col][row]);
        }
        fprintf(file, "\n\n");
    }
    fclose(file);
}

void saveVectors(double *leftVector, double *rightVector, int rank) {
    char fileName[40];
    FILE *file;
    sprintf(fileName, "result%d.txt", rank);

    file = fopen(fileName, "a");

    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }
    int row;
    fprintf(file, "\nleftVector:\n");
    for (row = 0; row < ROW; row++) {
        fprintf(file, "%1.3f | ", leftVector[row]);
    }
    fprintf(file, "\n\nrightVector:\n");
    for (row = 0; row < ROW; row++) {
        fprintf(file, "%1.3f | ", rightVector[row]);
    }
    fprintf(file, "\n");
    fclose(file);
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

void run(double** tab, int rank, MPI_Comm comm_cart) {

    double **prevTab = makeCopy(tab), rightVector[COL], leftVector[ROW];
    int i, row, col, right, left, direction = 0, disp = 1;

    MPI_Status recv_status, recv_status2;
    MPI_Request request, request2;

    MPI_Cart_shift(comm_cart, direction, disp, &left, &right);

    saveTable(tab, rank, 0, 1);
    for (i = 0; i < maxIter; i++) {
        saveVectors(tab[0], tab[COL - 1], rank);
        //            if (rank == 0) {printfVector(rightVector, rank);printfVector(leftVector, rank);}
        if (right >= 0 && right <= size) {
            MPI_Isend(tab[COL - 1], ROW, MPI_DOUBLE, right, TAG, comm_cart, &request);
            MPI_Recv(leftVector, ROW, MPI_DOUBLE, right, TAG, comm_cart, &recv_status);
        }

        if (left >= 0 && left <= size) {
            MPI_Isend(tab[0], ROW, MPI_DOUBLE, left, TAG, comm_cart, &request);
            MPI_Recv(rightVector, ROW, MPI_DOUBLE, left, TAG, comm_cart, &recv_status);
        }

        saveVectors(leftVector, rightVector, rank);
        for (row = 0; row < ROW; row++) {
            for (col = 0; col < COL; col++) {
                double left = 0, right = 0, top = 0, bottom = 0;
                if (row > 0) {
                    top = prevTab[col][row - 1];
                    //                                        if (rank == 0) {
                    //                                            printf("\n0-");
                    //                                        }
                }
                if (row < ROW - 1) {
                    bottom = prevTab[col][row + 1];
                    //                                        if (rank == 0) {
                    //                                            printf("1-");
                    //                                        }
                }
                if (col < COL - 1) {
                    right = prevTab[col + 1][row];
                    //                                        if (rank == 0) {
                    //                                            printf("2-");
                    //                                        }
                } else if (col == COL - 1 && rank < size - 1) {
                    right = leftVector[row];
                    //                                        if (rank == 0) {
                    //                                            printf("3-");
                    //                                        }
                }
                if (col > 0) {
                    left = prevTab[col - 1][row];
                    //                                        if (rank == 0) {
                    //                                            printf("4-");
                    //                                        }
                } else if (col == 0 && rank > 0) {
                    left = rightVector[row];
                    //                                        if (rank == 0) {
                    //                                            printf("5\n");
                    //                                        }
                }
                //                if (rank == 1) {
                //                    printf("\nrank:%d i:%d - [%d][%d], \n   [%1.3f]\n[%1.3f]  [%1.3f]\n   [%1.3f]\n", rank, i, col, row, top, left, right, bottom);
                //                }
                tab[col][row] = calculateValue(left, right, top, bottom, prevTab[col][row]);

            }
        }
        //        for (row = 0; row < ROW; row++) {
        //            for (col = 0; col < COL; col++) {
        //                prevTab[col][row] = tab[col][row];
        //            }
        //        }
        *prevTab = *tab;
        saveTable(tab, rank, i + 1, 0);
        //        break;
        //        printf("i:%d\n", i);
    }
    //    if (rank == 0) {printfTable(tab, rank);}

    //    if (rank == 0) {printfVector(rightVector, rank);printfVector(leftVector, rank);}

}

/*
 * Usage:
 *  ./pcam 100 100 0.001 0.01 500
 * 
 * Above arguments will initialize variables like below:
 * row = 100
 * col = 100
 * h = 0.001
 * dt = 0.01
 * maxIter = 500
 * 
 */
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 8) {
        ROW = DEFAULT_ROW;
        COL = DEFAULT_COL / size;
        h = DEFAULT_H;
        dt = DEFAULT_DT;
        maxIter = DEFAULT_MAX_ITER;
    } else {
        ROW = atof(argv[2]);
        COL = atof(argv[1]) / size;
        h = atof(argv[3]);
        dt = atof(argv[4]);
        maxIter = atof(argv[4]);
    }

    pow_h = h*h;
    start = MPI_Wtime();

    int ierror, my_rank, dims[2] = {size, 1}, periods[2] = {0, 0}, ndims = 2, reorder = 0;
    MPI_Comm comm_cart;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

    double** tab = initializeTable(tab, my_rank, size);

    run(tab, my_rank, comm_cart);

    printfTable(tab, my_rank);
    
    MPI_Finalize();

    end = MPI_Wtime();
    elapsed = end - start;
    printf("\n\n[%d] Elapsed time: %f\n", my_rank, elapsed);

    return (EXIT_SUCCESS);
}



