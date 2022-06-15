using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.Events;

public class SceneController : MonoBehaviour
{
	[Header("Texture settings")]
	public int Width = 256;
	public int Height = 256;

	[Header("Raycast settings")]
	public int MaxHits = 20;
	public int RayDist = 15;

	[Header("Event Listeners")]
	public UnityEvent OnProjectionsComplete;
	public UnityEvent OnExportComplete;

	static SceneController _instance;
	public static SceneController Settings => _instance;

	public SurfaceCountPlaneProjection[] _planes = new SurfaceCountPlaneProjection[3];
	string _savePath;

    private void Awake()
    {
        if (_instance != null)
        {
            Debug.LogWarning("Another instance of " + name + " is alredy loaded! Please check for possible errors");
        }
		_instance = this;
		Debug.Log("Settings instance set up");

		_savePath = Application.dataPath + "/Exports/";
		_planes = FindObjectsOfType<SurfaceCountPlaneProjection>();
	}

	public void ComputeProjections()
    {
        Debug.Log("Computing projections...");
        StartCoroutine(DoProjections());
    }

    public void ExportTextures()
    {
        var prefix = System.DateTime.Now.ToString("yyMMdd_hhmmss_");
        StartCoroutine(DoExport(prefix));
    }

    IEnumerator DoExport(string prefix)
    {
        yield return null;
        foreach (var plane in _planes)
        {
            var px = plane.Pixels;
            // flip png as pixels are ordered left to right, bottom to top
            System.Array.Reverse(px, 0, px.Length);
            for (int i = 0; i < Height; i++)
            {
                System.Array.Reverse(px, i * Height, Height);
            }
            var tx = new Texture2D(Width, Height, TextureFormat.RGB24, false);
            tx.SetPixels32(px);
            var png = tx.EncodeToPNG();
            var filename = prefix + plane.planeOrientation.ToString() + ".png";
            File.WriteAllBytes(_savePath + filename, png);
            Debug.Log(filename + " successfully saved to " + _savePath);
            yield return null;
        }
		OnExportComplete.Invoke();
    }

    IEnumerator DoProjections()
	{
        yield return null;
        foreach (var plane in _planes)
		{
			plane.ComputeProjections();
        }
		OnProjectionsComplete.Invoke();
        Debug.Log("...Done!");
    }

	public enum PlaneType
	{
		VERTICAL, HORIZONTAL, LATERAL
	}
}

// time.deltatime
