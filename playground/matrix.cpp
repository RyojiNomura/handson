#include <stdlib.h>
#include <iostream>

using namespace std;

int a[100][100];

int main(){
    for (int i=0; i<100; i++){
        for (int j=0; j<100; j++){
            a[i][j] = 1;
            cout << a[i][j] << endl;
        }
    }
    return 0;
}