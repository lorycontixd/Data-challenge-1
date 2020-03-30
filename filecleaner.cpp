#include <iostream>
#include <fstream>
#include <cstdlib>
#include <string>
#include <vector>
using namespace std;

int main(int argc, char** argv){
	if(argc!=4){
		cerr << "Program "<<argv[0]<<" requires <country> <filename_input> <filename_output> "<<endl;
		exit(EXIT_FAILURE);
	}
	string filename = argv[2]; //Load arguement as filename
	string filename2 = argv[3];
	string country = argv[1];
	ifstream myfile(filename); //Open file stream
	if(myfile.fail()){ //Check if stream was opened
		cerr << "Failed to load stream"<<endl;
		exit(EXIT_FAILURE);
	}
	string dummyLine; //Line for loading first line to be ignored
	getline(myfile,dummyLine); //Ignore first line
	
	vector<string> v; 
	string line; //String for saving the line
	string check = country; //Phrase to check
	size_t found; //Position of string found
	while(!myfile.eof()){
		getline(myfile,line);
		found = line.find(check);
		if(found!=std::string::npos){
			v.push_back(line);
		}
		//cout << line <<endl;
	}
	myfile.close();
	ofstream myfile2(filename2);
	if(myfile2.fail()){ //Check if stream was opened
		cerr << "Failed to load stream"<<endl;
		exit(EXIT_FAILURE);
	}
	for(unsigned i=0;i<v.size();i++){
		myfile2 << v[i] << endl;
		//cout << v[i]<<endl;
	}
	myfile2.close();
	cout << "Clean complete..."<<endl;
	exit(EXIT_SUCCESS);
}