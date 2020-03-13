#include<stdio.h>
#include<iostream>
#include<stack>
#include<cstdio>
#include<cstdlib>

using namespace std;

int solve(stack<int> &st, string &str){
    int b = st.top(); st.pop();
    int a = st.top(); st.pop();
    if(str=="+") return a + b;
    else if(str=="-") return a - b;
    else if(str=="*") return a * b;
    else if(str=="/") return a / b;    
}

int main(){
    stack <int> st;
    string str;
    while(1){
        cin >> str;
        if(str == "+" || str == "-" || str == "*" || str == "/" ) {
            st.push(solve(st, str));
        }else {
            int a = atoi(str.c_str());
            st.push(a);
        }
        if(getchar()=='\n'){
            cout << st.top() << endl;
           return 0;
        }
    }
}
