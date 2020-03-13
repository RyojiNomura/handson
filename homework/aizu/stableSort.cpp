#include<iostream>
#include<vector>
#include<string>
using namespace std;

struct card{char suit; int value;};

void trace(vector<struct card> &a, int &n){
    for(int i=0; i<n; i++){
        cout << a.at(i).suit << a.at(i).value;
        if(i<n-1){
            cout << " ";
        }
    }
    cout << endl;
}

void isSameOrder(vector<struct card> &a,vector<struct card> &b, int n){
    bool c = true;
    for(int i=0; i<n; i++){
        if (a.at(i).suit != b.at(i).suit){
            c = false;
            break;
        }
    }
    if(c){
        cout << "Stable" << endl;
    }else{
        cout << "Not stable" << endl;

    }
}

vector<struct card> bubbleSort(vector<struct card> a, int n){
    int cnt = 0;
    for(int i=0; i<n; i++){
        for(int j=0; j<n-1; j++){
            struct card vs = a.at(j);
            struct card ws = a.at(j+1);
            if ( vs.value > ws.value){
                a.at(j) = ws; 
                a.at(j+1) = vs;
                cnt++; 
            }
        }
    }
    return a;
}

vector<struct card> selectionSort(vector<struct card> a, int n){
    int cnt = 0;
    for(int i=0; i<n; i++){
        struct card min = a.at(i);
        int loc = i;
        for(int j=i; j<n; j++){
            struct card val = a.at(j);
            if(min.value > val.value){
                min = a.at(j);
                loc = j;
            }
        }
        if(loc != i){
            cnt++;
            a.at(loc) = a.at(i);
            a.at(i) = min;
        }
    }
    return a;
}

int main(){
    int n;
    cin >> n;
    vector<struct card> a,  bs, ss;
    for(int i=0; i<n; i++){
        string b;
        cin >> b;
        struct card c;
        c.suit = b.at(0);
        c.value = (int)b.at(1) - (int)'0';
        a.push_back(c);
    }

    bs = bubbleSort(a, n);
    trace(bs, n);
    isSameOrder(bs, bs, n);

    ss = selectionSort(a, n);
    trace(ss, n);
    isSameOrder(bs, ss, n);

}
