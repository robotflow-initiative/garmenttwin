using UnityEngine;
using Valve.VR;
using System.IO;
using System.Collections.Generic;
using UnityEngine.VFX;
using Newtonsoft.Json;
using FreeImageAPI;
using System.Linq;
using UnityEngine.UI;
using System.Collections;
using System;

public class Pose
{
    public float[] T = new float[3];
    public float[] Q = new float[4];
    public float[,] R = new float[3, 3];
}
public class CameraData
{
    public float[,] K;
    public float[,] dist;
    public float[] fxy;
    public float[] cxy;
    public Pose extrinsic;
}
public class WorldPose
{
    public float[,] R;
    public float[,] T;
}


public class Manager : MonoBehaviour
{
    public float scale = 0.3f;
    public string datasetPath = "D:/DataSet";
    public Text textMode;
    public Text textClip;
    public Text textFrame;


    enum Mode
    {
        World = 0,
        Clip = 1,
        Pose = 2
    }
    Mode currentMode;

    Mode CurrentMode
    {
        get
        {
            return currentMode;
        }
        set
        {
            while (value < 0)
                value += 3;
            value = (Mode)((int)value % 3);
            currentMode = value;
            cloth?.RemoveAttach(leftHand);
            cloth?.RemoveAttach(rightHand);
            textMode.text = $"当前模式：{currentMode}";
            switch (currentMode)
            {
                case Mode.World:
                    leftLine.gameObject.SetActive(false);
                    rightLine.gameObject.SetActive(false);
                    break;
                case Mode.Clip:
                    leftLine.gameObject.SetActive(true);
                    rightLine.gameObject.SetActive(true);
                    break;
                case Mode.Pose:
                    leftLine.gameObject.SetActive(true);
                    rightLine.gameObject.SetActive(true);
                    break;
            }
        }
    }

    int clipIndex = 0;
    int ClipIndex
    {
        get
        {
            return clipIndex;
        }
        set
        {
            if (value < 0 || value >= clipsPath.Length) return;
            clipIndex = value;
            NextClip();
        }
    }

    int frameIndex;
    int FrameIndex
    {
        get
        {
            return frameIndex;
        }
        set
        {
            frameIndex = value;
            NextFrame();
            textFrame.text = $"起点帧：{StartIndex}， 终点帧：{EndIndex}，\n当前帧：{FrameIndex}";
        }
    }
    int startIndex = -1;
    int StartIndex
    {
        get
        {
            return startIndex;
        }
        set
        {
            startIndex = Mathf.Clamp(value, 0, handPose.Count - 1);
            if (EndIndex <= startIndex)
                EndIndex = startIndex + 1;
            SetLine();
            textFrame.text = $"起点帧：{StartIndex}， 终点帧：{EndIndex}，\n当前帧：{FrameIndex}";
        }
    }
    int endIndex = int.MaxValue;
    int EndIndex
    {
        get
        {
            return endIndex;
        }
        set
        {
            endIndex = Mathf.Clamp(value, startIndex, handPose.Count - 1);
            SetLine();
            textFrame.text = $"起点帧：{StartIndex}， 终点帧：{EndIndex}，\n当前帧：{FrameIndex}";
        }
    }
    public string frameName => handPose[FrameIndex].Key;

    public Transform leftBoard;
    public Transform rightBoard;

    public Transform leftHandBoard;
    public Transform rightHandBoard;

    public Transform leftHand;
    public Transform rightHand;

    public static Manager instance;

    public int leftHandCamera;
    public int rightHandCamera;

    public ClothInstance cloth;

    public MeshFilter exportMesh;
    OBJExporter oBJExporter;
    private void Awake()
    {
        if (instance == null)
        {
            instance = this;
        }
    }
    Dictionary<string, CameraData> cameraDatas;
    WorldPose worldPose;
    List<KeyValuePair<string, Dictionary<string, Dictionary<string, Pose>>>> handPose;

    public Transform world;
    public VisualEffect VFXPrafab;
    Dictionary<string, Texture2D> colorTex = new();
    Dictionary<string, Texture2D> depthTex = new();
    Dictionary<string, VisualEffect> pointCloud = new();
    public LineRenderer leftLine;
    public LineRenderer rightLine;
    Dictionary<string, Dictionary<string, List<Vector3>>> pointPos = new();
    Dictionary<string, Dictionary<string, List<Quaternion>>> pointQua = new();
    Dictionary<string, List<bool>> pointGrab = new();
    public List<string> cameraName = new List<string>();

    string[] clipsPath;
    void Start()
    {
        oBJExporter = new OBJExporter();
        oBJExporter.applyPosition = false;
        oBJExporter.applyRotation = false;
        oBJExporter.applyScale = false;
        oBJExporter.generateMaterials = false;

        foreach (var item in cameraName)
        {
            colorTex.Add(item, new Texture2D(960, 540, TextureFormat.RGB24, false, true));
            depthTex.Add(item, new Texture2D(960, 540, TextureFormat.R16, false, true));
            pointCloud.Add(item, Instantiate(VFXPrafab));
            pointCloud[item].transform.parent = world;
            pointCloud[item].name = item;
        }
        pointGrab.Add("left_hand", new List<bool>());
        pointGrab.Add("right_hand", new List<bool>());
        //创建line和pose
        foreach (var item in cameraName)
        {
            Dictionary<string, List<Vector3>> cameraPos = new Dictionary<string, List<Vector3>>();
            cameraPos.Add("left_hand", new List<Vector3>());
            cameraPos.Add("right_hand", new List<Vector3>());
            pointPos.Add(item, cameraPos);

            Dictionary<string, List<Quaternion>> cameraQua = new Dictionary<string, List<Quaternion>>();
            cameraQua.Add("left_hand", new List<Quaternion>());
            cameraQua.Add("right_hand", new List<Quaternion>());
            pointQua.Add(item, cameraQua);
        }

        clipsPath = Directory.GetDirectories(datasetPath);
        ClipIndex = 0;
    }
    string currentClipPath;
    string currentClothName;

    IEnumerator ReLoadCloth()
    {
        if (cloth != null)
            DestroyImmediate(cloth.gameObject);
        ClothInstance sourceCloth = Resources.Load<ClothInstance>($"garment3d/{currentClothName}/max_retopo_ObiCloth");
        if (sourceCloth == null)
        {
            Debug.LogWarning($"Not Find Cloth: {currentClothName}");
            yield break;
        }
        cloth = Instantiate(sourceCloth);
        for (int i = 0; i < cloth.actor.blueprint.invMasses.Length; i++)
        {
            cloth.actor.blueprint.invMasses[i] = 1f / 0.001f;
        }
        for (int t = 0; t < 3; t++)
        {
            yield return new WaitForFixedUpdate();
        }
        SaveMesh(Path.Combine(currentClipPath, "rest.obj"));
    }
    void NextClip()
    {
        currentClipPath = clipsPath[ClipIndex];
        Debug.Log(currentClipPath);
        string clipName = Path.GetFileName(currentClipPath);
        string[] split = clipName.Split("_");
        currentClothName = $"{split[0]}_{int.Parse(split[1])}";
        Debug.Log(currentClothName);
        StartCoroutine(ReLoadCloth());
        //json读取
        string json = File.ReadAllText(Path.Combine(currentClipPath, "cameras_info.jsonl"));
        cameraDatas = JsonConvert.DeserializeObject<Dictionary<string, CameraData>>(json, new UnityJsonConverter());

        json = File.ReadAllText(Path.Combine(currentClipPath, "realworld_rot_trans.jsonl"));
        worldPose = JsonConvert.DeserializeObject<WorldPose>(json, new UnityJsonConverter());

        json = File.ReadAllText(Path.Combine(currentClipPath, "res2.json"));
        handPose = JsonConvert.DeserializeObject<Dictionary<string, Dictionary<string, Dictionary<string, Pose>>>>(json, new UnityJsonConverter()).ToList();


        ////正向填充空数据
        //for (int i = 0; i < handPose.Count; i++)
        //{
        //    var item = handPose[i].Value;
        //    foreach (var c in cameraName)
        //    {
        //        if (!item.ContainsKey(c))
        //            item.Add(c, new Dictionary<string, Pose>());

        //        if (!item[c].ContainsKey("left_hand"))
        //        {
        //            item[c].Add("left_hand", new Pose());
        //            if (i > 0)
        //                item[c]["left_hand"] = handPose[i - 1].Value[c]["left_hand"];
        //        }
        //        if (i > 0 && item[c]["left_hand"].T.Length < 3)
        //            item[c]["left_hand"] = handPose[i - 1].Value[c]["left_hand"];

        //        if (!item[c].ContainsKey("right_hand"))
        //        {
        //            item[c].Add("right_hand", new Pose());
        //            if (i > 0)
        //                item[c]["right_hand"] = handPose[i - 1].Value[c]["right_hand"];
        //        }
        //        if (i > 0 && item[c]["right_hand"].T.Length < 3)
        //            item[c]["right_hand"] = handPose[i - 1].Value[c]["right_hand"];
        //    }
        //}
        ////反向填充空数据
        //for (int i = handPose.Count - 1; i >= 0; i--)
        //{
        //    var item = handPose[i].Value;
        //    foreach (var c in cameraName)
        //    {
        //        if (!item[c].ContainsKey("left_hand"))
        //            item[c].Add("left_hand", handPose[i + 1].Value[c]["left_hand"]);
        //        if (item[c]["left_hand"].T.Length < 3)
        //            item[c]["left_hand"] = handPose[i + 1].Value[c]["left_hand"];

        //        if (!item[c].ContainsKey("right_hand"))
        //            item[c].Add("right_hand", handPose[i + 1].Value[c]["right_hand"]);
        //        if (item[c]["right_hand"].T.Length < 3)
        //            item[c]["right_hand"] = handPose[i + 1].Value[c]["right_hand"];
        //    }
        //}

        //获取line和pose数据
        foreach (var item in cameraName)
        {
            pointPos[item]["left_hand"] = handPose.Select((s) =>
            {
                Pose value = s.Value[item]["left_hand"];
                if (value.T.Length < 3)
                    return Vector3.zero;
                return new Vector3(value.T[0], -value.T[1], value.T[2]);
            }).ToList();
            pointPos[item]["right_hand"] = handPose.Select((s) =>
            {
                Pose value = s.Value[item]["right_hand"];
                if (value.T.Length < 3)
                    return Vector3.zero;
                return new Vector3(value.T[0], -value.T[1], value.T[2]);
            }).ToList();

            pointQua[item]["left_hand"] = handPose.Select((s) =>
            {
                Pose value = s.Value[item]["left_hand"];
                if (value.Q.Length < 4)
                    return Quaternion.identity;
                return new Quaternion(-value.Q[1], value.Q[2], -value.Q[3], value.Q[0]);
            }).ToList();
            pointQua[item]["right_hand"] = handPose.Select((s) =>
            {
                Pose value = s.Value[item]["right_hand"];
                if (value.Q.Length < 4)
                    return Quaternion.identity;
                return new Quaternion(-value.Q[1], value.Q[2], -value.Q[3], value.Q[0]);
            }).ToList();

            pointGrab["left_hand"] = handPose.Select((s) =>
            {
                return false;
            }).ToList();
            pointGrab["right_hand"] = handPose.Select((s) =>
            {
                return false;
            }).ToList();
        }

        textClip.text = $"当前片段：{clipsPath[ClipIndex]}";

        //world.transform.position = new Vector3(worldPose.T[0, 0], -worldPose.T[1, 0], worldPose.T[2, 0]);
        //Matrix4x4 matrix = new Matrix4x4();
        //for (int i = 0; i < 3; i++)
        //{
        //    for (int j = 0; j < 3; j++)
        //    {
        //        matrix[i, j] = worldPose.R[i, j];
        //    }
        //}
        //world.transform.rotation = new Quaternion(-matrix.rotation.x, matrix.rotation.y, -matrix.rotation.z, matrix.rotation.w);

        //点云参数设置
        foreach (var item in cameraName)
        {
            SetVFXProp(pointCloud[item], cameraDatas[item]);
        }
        CurrentMode = Mode.World;
        StartIndex = 0;
        EndIndex = handPose.Count - 1;
        FrameIndex = 0;
    }
    void SetLine()
    {
        SetLine(leftLine, pointPos[cameraName[leftHandCamera]]["left_hand"].Skip(StartIndex).Take(EndIndex - StartIndex + 1).ToArray());
        SetLine(rightLine, pointPos[cameraName[rightHandCamera]]["right_hand"].Skip(StartIndex).Take(EndIndex - StartIndex + 1).ToArray());
    }

    void SetLine(LineRenderer lineRenderer, Vector3[] vector3s)
    {
        lineRenderer.positionCount = vector3s.Length;
        lineRenderer.SetPositions(vector3s);
    }
    void NextFrame()
    {
        ReadImage(currentClipPath);
        SetBoardTrans();
    }

    public static void SetVFXProp(VisualEffect vfx, CameraData data)
    {
        Matrix4x4 v = Matrix4x4.TRS(new Vector3(data.extrinsic.T[0], -data.extrinsic.T[1], data.extrinsic.T[2]), new Quaternion(-data.extrinsic.Q[1], data.extrinsic.Q[2], -data.extrinsic.Q[3], data.extrinsic.Q[0]), Vector3.one);
        var fxy = new Vector2(data.fxy[0], data.fxy[1]);
        var cxy = new Vector2(data.cxy[0], data.cxy[1]);
        var k = new Vector3(data.dist[0, 0], data.dist[0, 1], data.dist[0, 4]);
        var p = new Vector2(data.dist[0, 2], data.dist[0, 3]);
        vfx.SetVector2("fxy", fxy);
        vfx.SetVector2("cxy", cxy);
        vfx.SetVector3("k1k2k3", k);
        vfx.SetVector2("p1p2", p);
        vfx.SetFloat("zscale", instance.scale * 65.535f);
        //vfx.SetMatrix4x4("Transform", v);
        vfx.transform.localPosition = v.GetPosition();
        vfx.transform.localRotation = v.rotation;
    }

    bool mode = false;

    void Update()
    {
        if (SteamVR_Input.GetStateDown("default", "Mode", SteamVR_Input_Sources.Any) || Input.GetKeyDown(KeyCode.Tab))
        {
            mode = true;
        }
        else if (SteamVR_Input.GetStateUp("default", "Mode", SteamVR_Input_Sources.Any) || Input.GetKeyUp(KeyCode.Tab))
        {
            mode = false;
        }

        if (mode && SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.LeftHand) || Input.GetKeyDown(KeyCode.S))
        {
            StartCoroutine(SaveClip());
        }
        if (mode && SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.RightHand) || Input.GetKeyDown(KeyCode.R))
        {
            StartCoroutine(ReLoadCloth());
        }
        if (mode && SteamVR_Input.GetStateDown("default", "GrabGrip", SteamVR_Input_Sources.LeftHand))
        {
            CurrentMode--;
        }
        if (mode && SteamVR_Input.GetStateDown("default", "GrabGrip", SteamVR_Input_Sources.RightHand))
        {
            CurrentMode++;
        }
        if (SteamVR_Input.GetStateDown("default", "SnapTurnLeft", SteamVR_Input_Sources.Any) || Input.GetKeyDown(KeyCode.LeftArrow))
        {
            if (mode)
            {
                FrameIndex = StartIndex;
            }
            else
            {
                Debug.Log("上一帧");
                FrameIndex--;
                if (coroutine == null)
                    coroutine = StartCoroutine(HighSpeedNextFrame(-1));
            }
        }
        else if (SteamVR_Input.GetStateDown("default", "SnapTurnRight", SteamVR_Input_Sources.Any) || Input.GetKeyDown(KeyCode.RightArrow))
        {
            if (mode)
            {
                FrameIndex = EndIndex;
            }
            else
            {
                Debug.Log("下一帧");
                FrameIndex++;
                if (coroutine == null)
                    coroutine = StartCoroutine(HighSpeedNextFrame(1));
            }
        }
        if (!mode && SteamVR_Input.GetStateUp("default", "SnapTurnLeft", SteamVR_Input_Sources.Any))
        {
            if (coroutine != null)
            {
                StopCoroutine(coroutine);
                coroutine = null;
            }
        }
        if (!mode && SteamVR_Input.GetStateUp("default", "SnapTurnRight", SteamVR_Input_Sources.Any))
        {
            if (coroutine != null)
            {
                StopCoroutine(coroutine);
                coroutine = null;
            }
        }
        if (mode && SteamVR_Input.GetStateDown("default", "TeleportUp", SteamVR_Input_Sources.Any) || Input.GetKeyDown(KeyCode.UpArrow))
        {
            Debug.Log("上一Video");
            ClipIndex--;
        }
        else if (mode && SteamVR_Input.GetStateDown("default", "TeleportDown", SteamVR_Input_Sources.Any) || Input.GetKeyDown(KeyCode.DownArrow))
        {
            Debug.Log("下一Video");
            ClipIndex++;
        }
        if (!mode)
            switch (CurrentMode)
            {
                case Mode.World:
                    if (SteamVR_Input.GetStateDown("default", "GrabGrip", SteamVR_Input_Sources.LeftHand))
                    {
                        world.SetParent(leftHand);
                    }
                    else if (SteamVR_Input.GetStateDown("default", "GrabGrip", SteamVR_Input_Sources.RightHand))
                    {
                        world.SetParent(rightHand);
                    }
                    else if (SteamVR_Input.GetStateUp("default", "GrabGrip", SteamVR_Input_Sources.Any))
                    {
                        world.SetParent(null);
                    }
                    if (SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.LeftHand))
                    {
                        cloth.AddAttach(leftHand);
                    }
                    if (SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.RightHand))
                    {
                        cloth.AddAttach(rightHand);
                    }
                    if (SteamVR_Input.GetStateUp("default", "InteractUI", SteamVR_Input_Sources.LeftHand))
                    {
                        cloth.RemoveAttach(leftHand);
                    }
                    if (SteamVR_Input.GetStateUp("default", "InteractUI", SteamVR_Input_Sources.RightHand))
                    {
                        cloth.RemoveAttach(rightHand);
                    }
                    break;
                case Mode.Clip:
                    if (SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.LeftHand) || Input.GetKeyDown(KeyCode.J))
                    {
                        StartIndex = FrameIndex;
                    }
                    else if (SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.RightHand) || Input.GetKeyDown(KeyCode.K))
                    {
                        EndIndex = FrameIndex;
                    }
                    break;
                case Mode.Pose:
                    if (SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.LeftHand))
                    {
                        pointPos[cameraName[leftHandCamera]]["left_hand"][FrameIndex] = world.InverseTransformPoint(leftHandBoard.position);
                        pointQua[cameraName[leftHandCamera]]["left_hand"][FrameIndex] = Quaternion.Inverse(world.rotation) * leftHandBoard.rotation;
                        SetBoardTrans();
                    }
                    if (SteamVR_Input.GetStateDown("default", "InteractUI", SteamVR_Input_Sources.RightHand))
                    {
                        pointPos[cameraName[rightHandCamera]]["right_hand"][FrameIndex] = world.InverseTransformPoint(rightHandBoard.position);
                        pointQua[cameraName[rightHandCamera]]["right_hand"][FrameIndex] = Quaternion.Inverse(world.rotation) * rightHandBoard.rotation;
                        SetBoardTrans();
                    }


                    if (SteamVR_Input.GetStateDown("default", "GrabGrip", SteamVR_Input_Sources.LeftHand) || Input.GetKeyDown(KeyCode.N))
                    {
                        bool grip = pointGrab["left_hand"][FrameIndex];
                        for (int i = FrameIndex; i < pointGrab["left_hand"].Count; i++)
                        {
                            if (grip != pointGrab["left_hand"][i]) break;
                            pointGrab["left_hand"][i] = !pointGrab["left_hand"][i];
                        }
                        SetBoardTrans();
                    }
                    if (SteamVR_Input.GetStateDown("default", "GrabGrip", SteamVR_Input_Sources.RightHand) || Input.GetKeyDown(KeyCode.M))
                    {
                        bool grip = pointGrab["right_hand"][FrameIndex];
                        for (int i = FrameIndex; i < pointGrab["right_hand"].Count; i++)
                        {
                            if (grip != pointGrab["right_hand"][i]) break;
                            pointGrab["right_hand"][i] = !pointGrab["right_hand"][i];
                        }
                        SetBoardTrans();
                    }
                    break;
            }
    }

    Coroutine coroutine = null;
    IEnumerator HighSpeedNextFrame(int i)
    {
        yield return new WaitForSeconds(0.5f);
        while (true)
        {
            Debug.Log("开始加速切换");
            FrameIndex += i;
            yield return new WaitForSeconds(0.1f);
        }
    }
    private void UpdataClothGrip()
    {
        if (!pointGrab["left_hand"][FrameIndex])
            cloth.RemoveAttach(leftBoard);
        else
            cloth.AddAttach(leftBoard);
        if (!pointGrab["right_hand"][FrameIndex])
            cloth.RemoveAttach(rightBoard);
        else
            cloth.AddAttach(rightBoard);
        //{
        //    if (pointGrab["left_hand"][FrameIndex] && !pointGrab["left_hand"][FrameIndex - 1])
        //        cloth.AddAttach(leftBoard);
        //    if (pointGrab["right_hand"][FrameIndex] && !pointGrab["right_hand"][FrameIndex - 1])
        //        cloth.AddAttach(rightBoard);
        //}
    }
    IEnumerator SaveClip()
    {
        string clipPath = Path.Combine(currentClipPath, $"SaveClip_{handPose[StartIndex].Key}_{handPose[EndIndex].Key}");
        Directory.CreateDirectory(clipPath);
        for (int i = StartIndex; i <= EndIndex; i++)
        {
            FrameIndex = i;
            UpdataClothGrip();
            for (int t = 0; t < 10; t++)
            {
                yield return new WaitForFixedUpdate();
            }
            SaveMesh(Path.Combine(clipPath, $"ClothMesh_{handPose[FrameIndex].Key}.obj"));
        }

        Dictionary<string, Dictionary<string, Tuple<float[], float[], bool>>> handData = new();
        for (int i = StartIndex; i <= EndIndex; i++)
        {
            Vector3 pos = pointPos[cameraName[leftHandCamera]]["left_hand"][i];
            Quaternion qua = pointQua[cameraName[leftHandCamera]]["left_hand"][i];
            bool grab = pointGrab["left_hand"][i];
            float[] posf = new float[] { pos.x, -pos.y, pos.z };
            float[] quaf = new float[] { qua.w, -qua.x, qua.y, -qua.z };
            Tuple<float[], float[], bool> leftHandData = new Tuple<float[], float[], bool>(posf, quaf, grab);

            pos = pointPos[cameraName[rightHandCamera]]["right_hand"][i];
            qua = pointQua[cameraName[rightHandCamera]]["right_hand"][i];
            grab = pointGrab["right_hand"][i];
            posf = new float[] { pos.x, -pos.y, pos.z };
            quaf = new float[] { qua.w, -qua.x, qua.y, -qua.z };
            Tuple<float[], float[], bool> rightHandData = new Tuple<float[], float[], bool>(posf, quaf, grab);

            handData.Add(handPose[i].Key, new Dictionary<string, Tuple<float[], float[], bool>>());
            handData[handPose[i].Key].Add("left_hand", leftHandData);
            handData[handPose[i].Key].Add("right_hand", rightHandData);
        }
        string jsonData = JsonConvert.SerializeObject(handData);
        File.WriteAllText(Path.Combine(clipPath, $"HandPose_{handPose[StartIndex].Key}_{handPose[EndIndex].Key}.json"), jsonData);
        yield break;
    }
    void SaveMesh(string path)
    {
        Mesh mesh = new Mesh();
        mesh.vertices = cloth.GetComponentInChildren<MeshFilter>().sharedMesh.vertices;
        mesh.triangles = cloth.GetComponentInChildren<MeshFilter>().sharedMesh.triangles;

        Span<Vector3> temp1 = new Span<Vector3>(mesh.vertices);
        Span<Vector3> temp2 = new Span<Vector3>(mesh.vertices);
        cloth.GetComponentInChildren<MeshFilter>().transform.TransformPoints(mesh.vertices, temp1);
        exportMesh.transform.InverseTransformPoints(temp1, temp2);
        mesh.vertices = temp2.ToArray();

        mesh.RecalculateNormals();

        exportMesh.mesh = mesh;
        oBJExporter.Export(new GameObject[] { exportMesh.gameObject }, path);
    }
    void SetBoardTrans()
    {
        leftBoard.localPosition = pointPos[cameraName[leftHandCamera]]["left_hand"][FrameIndex];
        leftBoard.localRotation = pointQua[cameraName[leftHandCamera]]["left_hand"][FrameIndex];
        if (pointGrab["left_hand"][FrameIndex])
        {
            leftBoard.Find("Quad").gameObject.SetActive(true);
            leftBoard.Find("Sphere").gameObject.SetActive(false);
        }
        else
        {
            leftBoard.Find("Quad").gameObject.SetActive(false);
            leftBoard.Find("Sphere").gameObject.SetActive(true);
        }

        rightBoard.localPosition = pointPos[cameraName[rightHandCamera]]["right_hand"][FrameIndex];
        rightBoard.localRotation = pointQua[cameraName[rightHandCamera]]["right_hand"][FrameIndex];
        if (pointGrab["right_hand"][FrameIndex])
        {
            rightBoard.Find("Quad").gameObject.SetActive(true);
            rightBoard.Find("Sphere").gameObject.SetActive(false);
        }
        else
        {
            rightBoard.Find("Quad").gameObject.SetActive(false);
            rightBoard.Find("Sphere").gameObject.SetActive(true);
        }
        SetLine();
    }

    private void ReadImage(string path)
    {
        foreach (var item in cameraName)
        {
            byte[] pngBytes;
            string imagePath;
            imagePath = Path.Combine(path, item, "color", frameName + ".png");
            if (!File.Exists(imagePath))
            {
                pointCloud[item].gameObject.SetActive(false);
                continue;
            }
            else
                pointCloud[item].gameObject.SetActive(true);

            pngBytes = File.ReadAllBytes(imagePath);
            colorTex[item].LoadImage(pngBytes);
            colorTex[item].Apply();

            imagePath = Path.Combine(path, item, "depth", frameName + ".png");
            if (!File.Exists(imagePath))
            {
                pointCloud[item].gameObject.SetActive(false);
                continue;
            }
            else
                pointCloud[item].gameObject.SetActive(true);

            FIBITMAP bitmap = FreeImage.Load(FREE_IMAGE_FORMAT.FIF_PNG, imagePath, FREE_IMAGE_LOAD_FLAGS.DEFAULT);

            depthTex[item] = ConvertFIBITMAPToTexture2D(bitmap);
            //pngBytes = File.ReadAllBytes(imagePath);
            //depthTex[i].LoadImage(pngBytes);
            //depthTex[i].Apply();

            pointCloud[item].SetTexture("Color", colorTex[item]);
            pointCloud[item].SetTexture("Depth", depthTex[item]);

            pointCloud[item].Reinit();
            pointCloud[item].Play();
        }
    }
    private Texture2D ConvertFIBITMAPToTexture2D(FIBITMAP bitmap)
    {
        int width = (int)FreeImage.GetWidth(bitmap);
        int height = (int)FreeImage.GetHeight(bitmap);

        // Create a new Texture2D
        Texture2D texture = new Texture2D(width, height, TextureFormat.R16, false);

        // Lock the texture to get access to its pixel data
        texture.LoadRawTextureData(FreeImage.GetBits(bitmap), width * height * 2);
        texture.Apply();

        return texture;
    }
}
