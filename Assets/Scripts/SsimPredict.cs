using System.Collections;
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

    Tensor ComputeProjections() 
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
                for(int j=0; j<width; j++)
                    projections[0, i, j, k] = px[i*width + j][0];
        }

        return projections;
    }

    Tensor ComputeCnnFeatures()
    {
        var model = ModelLoader.Load(cnnModelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        var inputTensor = ComputeProjections();
        worker.Execute(inputTensor);
        var cnnFeatures = worker.PeekOutput();
        inputTensor.Dispose();
        worker.Dispose();

        return cnnFeatures;
    }

    Tensor PredictSsim()
    {
        var model = ModelLoader.Load(ffnModelSource);
        var worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, model);
        var inputTensor = ComputeCnnFeatures();
        worker.Execute(inputTensor);
        var output = worker.PeekOutput();
        inputTensor.Dispose();
        worker.Dispose();

        return output;
    }

    void Start()
    {
        cam = this.GetComponent<Camera>();
        projectionBox.SetActive(false);

        print(lodContainer.transform);
    }

    // Update is called once per frame
    void Update()
    {
        var pos = cam.transform.position;
        var velocity = cam.velocity;
        var pos2 = pos + velocity * 0.5f;
    }
}
