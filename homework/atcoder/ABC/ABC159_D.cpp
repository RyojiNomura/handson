#include <bits/stdc++.h>
#include<stdio.h>
using namespace std;

#define MAX 200000
int a[MAX];

int num(int k, int n){
    int i, j, cnt=0;
    for(i=0; i<n; i++){
        if(i==k) continue;
        for(j=i+1; j<n; j++){
            if(j==k) continue;
            if(a[i] == a[j]) cnt++;
        }
    }
    return cnt;
}

int main(){
    int n,i;
    cin >> n;
    for (i=0; i<n; i++) cin >>a[i];
    for (i=0; i<n; i++){
        int ans = num(i, n);
        printf("%d\n", ans);
    }
}