pipeline {
    agent any
    // parameters {
    //     choice(name: 'problem_name', choices: ['houses', 'groceries', 'iris'], description: 'Choose the problem name')
    //     string(name: 'ml_pipeline_params_name', defaultValue: 'default', description: 'Specify the ml_pipeline_params file')
    //     string(name: 'feature_set_name', defaultValue: 'default', description: 'Specify the feature_set name/file')
    //     string(name: 'algorithm_name', defaultValue: 'default', description: 'Specify the algorithm (overrides problem_params)')
    //     string(name: 'algorithm_params_name', defaultValue: 'default', description: 'Specify the algorithm params')
    // }
    triggers {
        // Poll SCM every minute for new changes
        pollSCM('* * * * *')
    }
    options {
       // add timestamps to output
       timestamps()
    }
    environment { 
        MLFLOW_TRACKING_URL = 'http://mlflow:5000'
        MLFLOW_S3_ENDPOINT_URL = 'http://minio:9000'
        AWS_ACCESS_KEY_ID = "${env.ACCESS_KEY}"
        AWS_SECRET_ACCESS_KEY = "${env.SECRET_KEY}"

    }
    stages {
        stage('Install dependencies') {
            steps {
                sh 'make clean'
                sh 'make create_dir'
                sh 'make env'
                sh 'make install'
                sh 'echo Olá pessoal'
            }
        }
        stage('Run tests') {
            steps {
                sh 'make lint'
                sh 'make test'
            }
        }
        stage('Run pipeline') {
            steps {
                // sh 'python3 run_python_script.py pipeline ${problem_name} ${ml_pipeline_params_name} ${feature_set_name} ${algorithm_name} ${algorithm_params_name}'
                sh 'make run'
            }
       }
       stage('Production - Register Model and Acceptance Test') {
        //    when {
        //       allOf {
        //             equals expected: 'default', actual: "${params.ml_pipeline_params_name}"
        //             equals expected: 'default', actual: "${params.feature_set_name}"
        //             equals expected: 'default', actual: "${params.algorithm_name}"
        //             equals expected: 'default', actual: "${params.algorithm_params_name}"
        //        }
        //    }
           steps {
                // sh 'python3 run_python_script.py acceptance'
                sh 'echo production'
           }
        //    post {
        //         success {
        //             sh 'python3 run_python_script.py register_model ${MLFLOW_TRACKING_URL} yes'
        //             sh 'make up'
        //         }
        //         failure {
        //             sh 'python3 run_python_script.py register_model ${MLFLOW_TRACKING_URL} no'
        //         }
        //    }
       }
       stage('Experiment - Register Model and Acceptance Test') {
        //     when {
        //        anyOf {
        //             not { equals expected: 'default', actual: "${params.ml_pipeline_params_name}" }
        //             not { equals expected: 'default', actual: "${params.feature_set_name}"}
        //             not { equals expected: 'default', actual: "${params.algorithm_name}"}
        //             not { equals expected: 'default', actual: "${params.algorithm_params_name}"}
        //        }
        //    }
           steps {
                // sh '''
                // set +e
                // python3 run_python_script.py acceptance
                // set -e
                // '''
                // sh 'python3 run_python_script.py register_model ${MLFLOW_TRACKING_URL} no'
                sh 'echo run dev'
           }
       }
    }
}
