/* 
 * Copyright (C) 2017 NEC Laboratories America, Inc. ("NECLA"). All rights reserved.
 *
 * This source code is licensed under the license found in the LICENSE file in
 * the root directory of this source tree. An additional grant of patent rights
 * can be found in the PATENTS file in the same directory.
 */
#include<iostream>
#include<assert.h>
#include<mpi.h>

int main(int argc, char* argv[]){
    // assume 4 processes
    int* sendlist = new int[2];
    int* recvlist = new int[2];
    int* buf = new int[4];
    int rank;
    int nProc;
    int size=4;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProc);
    sendlist[0] = (rank+1)%nProc;
    recvlist[0] = (rank+3)%nProc;
    sendlist[1] = (rank+3)%nProc;
    recvlist[1] = (rank+1)%nProc;

    for(int i=0; i<4; i++) buf[i]=rank;
    printf("rank:%d sendlist: %d %d, recvlist: %d %d\n",rank, sendlist[0],sendlist[1], recvlist[0], recvlist[1]);
    int q=2;
    MPI_Group fromgroup, togroup, groupworld;
    MPI_Comm_group(MPI_COMM_WORLD, &groupworld);
    MPI_Group_incl(groupworld, q, recvlist, &fromgroup);
    MPI_Group_incl(groupworld, q, sendlist, &togroup);
    //add segment
    MPI_Win win;
    MPI_Win_create((void*)buf, size, sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_post(fromgroup, 0, win);
    printf("rank:%d ===add segment===",rank);
    for(int i=0; i<size; i++)
        printf("%d ",buf[i]);
    printf("\n");
    //push
    MPI_Win_start(togroup, 0, win); 
    for(int i=0; i<q; i++)
        MPI_Put((void*)(buf+rank), 1, MPI_INT, sendlist[i], rank, 1, MPI_INT, win);
    MPI_Win_complete(win);
    //reduce
    MPI_Win_wait(win);
    printf("rank:%d =====push=====",rank);
    for(int i=0; i<size; i++)
        printf("%d ",buf[i]);
    printf("\n");
    int sum=0;
    for(int i=0; i<size; i++) 
        sum+=buf[i];
    for(int i=0; i<size; i++) 
        buf[i]=sum/size;
    MPI_Barrier(MPI_COMM_WORLD);
    printf("rank:%d ====reduce====",rank);
    for(int i=0; i<size; i++)
        printf("%d ",buf[i]);
    printf("\n");

    MPI_Win_post(fromgroup, 0, win);// ready for incoming nodes to modify rbuf
    if(buf) delete[] buf;
    if(sendlist) delete[] sendlist;
    if(recvlist) delete[] recvlist;
    MPI_Win_free(&win);
    MPI_Group_free(&groupworld);
    MPI_Group_free(&fromgroup);
    MPI_Group_free(&togroup);
    MPI_Finalize();
    return 0;
}
