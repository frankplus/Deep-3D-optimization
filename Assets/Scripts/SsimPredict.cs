﻿using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using static SceneController;

public class SsimPredict : MonoBehaviour
{

    public NNModel cnnModelSource;
    public NNModel ffnModelSource;
    public GameObject projectionBox;
    private Camera cam;
    public GameObject lodContainer;

    private Tensor cnnFeatures;

    private Tensor ComputeProjections() 
    {
        var height = Settings.Height;
        var width = Settings.Width;

        SurfaceCountPlaneProjection[] planes = FindObjectsOfType<SurfaceCountPlaneProjection>();
        foreach (var plane in planes)
			plane.ComputeProjections();

        var projections = new Tensor(1, height, width, planes.Length);
        for(int k=0; k<planes.Length; k++)
        {
            var px = planes[k].Pixels;

            // flip png as pixels are ordered left to right, bottom to top
            System.Array.Reverse(px, 0, px.Length);
            for (int i = 0; i < height; i++)
                System.Array.Reverse(px, i * height, height);

            for(int i=0; i<height; i++)
                for(int j=0; j<width; j++) {
                    projections[0, i, j, k] = (float) px[i*width + j][0] / 255.0f;
                    // if (projections[0, i, j, k] != 1.0f)
                    //     print(projections[0, i, j, k]);
                }
        }

        return projections;
    }

    private Tensor ComputeCnnFeatures()
    {
        var model = ModelLoader.Load(cnnModelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        var inputTensor = ComputeProjections();
        worker.Execute(inputTensor);
        var features = worker.PeekOutput().DeepCopy();
        inputTensor.Dispose();
        worker.Dispose();

        return features;
    }

    private float PredictSsim(Vector3 posRef, Vector3 pos, float lodId)
    {
        float[] parameters = new float[] {pos.x, pos.y, pos.z, posRef.x, posRef.y, posRef.z, lodId};
        float[] cnnFeaturesArray = cnnFeatures.ToReadOnlyArray();

        // concatenate
        int batchSize = 1;
        float[] inputArray = new float[batchSize * (parameters.Length + cnnFeaturesArray.Length)];
        parameters.CopyTo(inputArray, 0);
        cnnFeaturesArray.CopyTo(inputArray, parameters.Length);
        Tensor inputTensor = new Tensor(batchSize, 
                                        1, 
                                        1, 
                                        parameters.Length + cnnFeaturesArray.Length, 
                                        inputArray, 
                                        "inputTensor"
                                        );

        var model = ModelLoader.Load(ffnModelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        worker.Execute(inputTensor);
        Tensor output = worker.PeekOutput();
        float ssim = output[0,0,0,0];
        output.Dispose();
        inputTensor.Dispose();
        worker.Dispose();

        return ssim;
    }

    void Start()
    {
        cam = this.GetComponent<Camera>();
        cnnFeatures = ComputeCnnFeatures();
        projectionBox.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        var posRef = cam.transform.position;
        var velocity = cam.velocity;
        var pos = posRef + velocity * 0.5f;
        float output = PredictSsim(posRef, pos, 1.0f);
        print("output: " + output);
    }

    void OnDestroy()
    {
        cnnFeatures.Dispose();
    }
}
