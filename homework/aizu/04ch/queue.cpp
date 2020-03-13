#include<iostream>
#include<vector>
#include<queue>
#include<string>
using namespace std;

struct job{string name; int time;};

int main(){
    int n, qtime;
    queue<struct job> jlist;
    queue<struct job> jdone;

    cin >> n >> qtime;
    for(int i=0; i<n; i++){
        string jname;
        int jtime;
        struct job jtemp;
        cin >> jname >> jtime;
        jtemp.name = jname;
        jtemp.time = jtime;
        jlist.push(jtemp);
    }
    int etime = 0;
    while(jlist.size() > 0){
        struct job jnow = jlist.front();
        jlist.pop();
        struct job jfin;
        int overtime = jnow.time - qtime;
        if(overtime > 0){
            etime += qtime;
            jnow.time = overtime;
            jlist.push(jnow);
        }else{
            etime += jnow.time;
            jfin.name = jnow.name;
            jfin.time = etime;
            jdone.push(jfin);
        }
    }

    for (int i=0; i<n; i++){
        struct job jtemp = jdone.front();
        jdone.pop();
        cout << jtemp.name << " " << jtemp.time << endl;
    }


}