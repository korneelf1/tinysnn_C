#include <stdio.h>
#include "softmax.h"
int findMaxIndex(float arr[], int size) {
    int maxIndex = 0;
    int i;
    // printf("Size: %d\n", size);
    for (int i = 0; i < size; i++) {
        // printf("Index: %d\n", i);
        // printf(" Value: %f\n", arr[i]);
        if (arr[i] > arr[maxIndex]) {
            maxIndex = i;
        }
    }
    // printf("Maximum index: %d\n", maxIndex);

    return maxIndex;
}
