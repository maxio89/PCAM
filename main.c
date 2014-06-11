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
#include <mpi.h>
//#include <openmpi-x86_64/mpi.h>
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

    double **prevTab = makeCopy(tab), rightVector[ROW], leftVector[ROW], receiving_avg=0.0, sending_avg=0.0, calculating_avg=0.0;
    int i, row, col, right, left, direction = 0, disp = 1;

    MPI_Status recv_status, recv_status2;
    MPI_Request request, request2, recv_request, recv_request2;

    MPI_Cart_shift(comm_cart, direction, disp, &left, &right);
    //printf("[%d] left %d right %d\n", rank, left, right);
    //    saveTable(tab, rank, 0, 1);
    for (i = 0; i < maxIter; i++) {
        //        saveVectors(tab[0], tab[COL - 1], rank);
        double a = 0.0, b, c = 0.0, d, sending_start, receiving_start, calculating_start, sending_end, receiving_end, calculating_end;
        
        
        calculating_start=MPI_Wtime();
        //        if (right >= 0 && right <= size) {
        //            MPI_Isend(&a, 1, MPI_DOUBLE, right, TAG, comm_cart, &request);
        //            MPI_Recv(&b, 1, MPI_DOUBLE, right, TAG, comm_cart, &recv_status);
        //        }
        //
        //        if (left >= 0 && left <= size) {
        //            MPI_Isend(&c, 1, MPI_DOUBLE, left, TAG, comm_cart, &request);
        //            MPI_Recv(&d, 1, MPI_DOUBLE, left, TAG, comm_cart, &recv_status);
        //        }

        //        saveVectors(leftVector, rightVector, rank);
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
                } else if (col == COL - 1 && rank < size - 1) {
                    right = leftVector[row];
                }
                if (col > 0) {
                    left = prevTab[col - 1][row];
                } else if (col == 0 && rank > 0) {
                    left = rightVector[row];
                }
                tab[col][row] = calculateValue(left, right, top, bottom, prevTab[col][row]);
            }
        }
        //        printf("i:%d\n", i);
        *prevTab = *tab;
        calculating_end = MPI_Wtime();
        
        sending_start = MPI_Wtime();
        if (right >= 0 && right <= size) {
            MPI_Isend(&tab[COL - 1], ROW, MPI_DOUBLE, right, TAG, comm_cart, &request);
            //            MPI_Irecv(leftVector, ROW, MPI_DOUBLE, right, TAG, comm_cart, &recv_request);
            //            MPI_Recv(leftVector, ROW, MPI_DOUBLE, right, TAG, comm_cart, &recv_status);
            //            printf("entered 1 %d\n", rank);
        }

        if (left >= 0 && left <= size) {
            MPI_Isend(&tab[0], ROW, MPI_DOUBLE, left, TAG, comm_cart, &request2);
            //            printfVector(tab[0],rank);
            //            MPI_Irecv(rightVector, ROW, MPI_DOUBLE, left, TAG, comm_cart, &recv_request2);
            //            MPI_Recv(rightVector, ROW, MPI_DOUBLE, left, TAG, comm_cart, &recv_status2);
            //            printf("entered 2 %d\n", rank);
        }
        sending_end = MPI_Wtime();
        receiving_start = sending_end;
        if (right >= 0 && right <= size) {

            MPI_Recv(leftVector, ROW, MPI_DOUBLE, right, TAG, comm_cart, &recv_status);
            //            MPI_Wait(&recv_request, &recv_status);
        }

        if (left >= 0 && left <= size) {

            MPI_Recv(rightVector, ROW, MPI_DOUBLE, left, TAG, comm_cart, &recv_status2);

        }
        receiving_end = MPI_Wtime(); 
        
        receiving_avg=receiving_avg+receiving_end-receiving_start;
        sending_avg=sending_avg+sending_end-sending_start;
        calculating_avg=calculating_avg+calculating_end-calculating_start;
//        printf("i:%d, row:%d, col:%d\n", i, row, col);
        printf("[%d] sending:%1.6fs, receiving:%1.6fs, calculating:%1.6fs \n", rank, sending_end-sending_start, receiving_end-receiving_start, calculating_end-calculating_start);
        //        MPI_Wait(&recv_request, &recv_status);
        //        MPI_Wait(&recv_request2, &recv_status2);
    }
    
    printf("[%d] sending_avg:%1.6fs, receiving_avg:%1.6fs, calculating_avg:%1.6fs \n", rank, sending_avg/i, receiving_avg/i, calculating_avg/i);
    //    saveTable(tab, rank, i + 1, 0);
}

/*
 * Usage:
 * mpirun -n 4 ./pcam 100 100 0.1 0.001 500
 * 
 * Above arguments will initialize variables like below:
 * row = 100
 * col = 100
 * h = 0.1
 * dt = 0.001
 * maxIter = 500
 * 
 */
int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 6) {
        ROW = DEFAULT_ROW;
        COL = DEFAULT_COL / size;
        h = DEFAULT_H;
        dt = DEFAULT_DT;
        maxIter = DEFAULT_MAX_ITER;
    } else {
        ROW = atoi(argv[1]);
        COL = atoi(argv[2]) / size;
        h = atof(argv[3]);
        dt = atof(argv[4]);
        maxIter = atoi(argv[5]);
    }
    printf("ROW:%d, COL:%d, h:%1.3f, dt:%1.3f, maxIter:%d, argc:%d\n", ROW, COL, h, dt, maxIter, argc);
    //    
    pow_h = h*h;
    start = MPI_Wtime();
    //
    int ierror, my_rank, dims[2] = {size, 1}, periods[2] = {0, 0}, ndims = 2, reorder = 0;
    MPI_Comm comm_cart;

    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);

    double** tab = initializeTable(tab, my_rank, size);

    run(tab, my_rank, comm_cart);

    //    printfTable(tab, my_rank);

    end = MPI_Wtime();
    elapsed = end - start;
    printf("\n[%d] Elapsed time: %f\n", my_rank, elapsed);

    MPI_Finalize();

    return (EXIT_SUCCESS);
}



