using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using static SceneController;

public class SurfaceCountPlaneProjection : MonoBehaviour
{
	public PlaneType planeOrientation;

	Renderer _renderer;
	Texture2D _texture;
	public Color32[] Pixels => _texture.GetPixels32();

	const int MAX_HITS = 20;
	const int RAY_DIST = 15;

	private void Start()
	{
		_renderer = GetComponent<Renderer>();
	}

	public void ComputeProjections()
    {
		_texture = new Texture2D(Settings.Width, Settings.Height, TextureFormat.RGB24, false);
		_renderer.material.mainTexture = _texture;

		var pixelSize = transform.right * Vector3.Dot(_renderer.bounds.size, transform.right) / _texture.width +
			transform.forward * Vector3.Dot(_renderer.bounds.size, transform.forward) / _texture.height;

		// get the center of the pixel adding half the pixel size
		var blPixelWorldPos = _renderer.bounds.max - pixelSize / 2;

		for (int u = 0; u < _texture.height; u++)
		{
			for (int v = 0; v < _texture.width; v++)
			{
				var currentPointDistance = u * Vector3.Dot(pixelSize, transform.right) * transform.right + v * Vector3.Dot(pixelSize, transform.forward) * transform.forward;
				var point = blPixelWorldPos - currentPointDistance;

				var hits = MultiHitRay(point);

				var intensity = (float)(MAX_HITS - hits) / MAX_HITS;
				_texture.SetPixel(u, v, new Color(intensity, intensity, intensity));
			}
		}
		_texture.Apply();
	}

	int HitRay(Vector3 point)
	{
		RaycastHit[] results = new RaycastHit[MAX_HITS];
		int hits = Physics.RaycastNonAlloc(point, transform.up, results, RAY_DIST);

		// do the return check to get all the "backfaces" which the first raycast direction will miss
		hits += Physics.RaycastNonAlloc(point + transform.up * (RAY_DIST + .01f), -transform.up, results, RAY_DIST);

		return hits;
	}

	int MultiHitRay(Vector3 point)
	{
		var rayCount = 0;
		var posList = new List<Vector3>();

		Vector3 hitPoint = point;

		while (Physics.Raycast(hitPoint, transform.up, out RaycastHit hit, RAY_DIST))
		{
			posList.Add(hitPoint);
			rayCount++;
			hitPoint = hit.point + (transform.up / 100.0f); // move a bit forward to pass beyond the collider
		}

		hitPoint = point + transform.up * RAY_DIST;

		while (Physics.Raycast(hitPoint, -transform.up, out RaycastHit hit, RAY_DIST))
		{
			posList.Add(hitPoint);
			rayCount++;
			hitPoint = hit.point + (-transform.up / 100.0f); // move a bit forward to pass beyond the collider
		}

		return rayCount;
	}
}
