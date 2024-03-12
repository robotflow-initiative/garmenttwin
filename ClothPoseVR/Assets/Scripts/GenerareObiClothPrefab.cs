using Obi;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;

public class GenerareObiClothPrefab : MonoBehaviour
{
    [MenuItem("RFUniverse/Generare ObiCloth Prefab")]
    public static void Generate()
    {
        ObiSolver stencil = null;
        List<GameObject> objs = Selection.GetFiltered<GameObject>(SelectionMode.Assets).ToList();
        foreach (var item in objs)
        {
            stencil = item.GetComponent<ObiSolver>();
            if (stencil != null)
            {
                objs.Remove(item);
                break;
            }
        }
        if (stencil == null)
        {
            Debug.LogWarning("Not Find ObiSolver Stencil in Selection");
            return;
        }
        Debug.Log(objs.Count);
        ObiSolver solver = Instantiate(stencil);
        foreach (var item in objs)
        {
            string file = AssetDatabase.GetAssetPath(item);
            string path = Path.GetDirectoryName(file);
            string name = Path.GetFileNameWithoutExtension(file);
            string blueprintPath = $"{path}/{name}_blueprint.asset";
            string prefabPath = $"{path}/{name}_ObiCloth.prefab";
            if (!string.IsNullOrEmpty(AssetDatabase.AssetPathToGUID(blueprintPath)) && !string.IsNullOrEmpty(AssetDatabase.AssetPathToGUID(prefabPath)))
                continue;
            Debug.Log(path);
            Debug.Log(name);

            GameObject prefab = Instantiate(item);
            ObiCloth cloth = solver.GetComponentInChildren<ObiCloth>();

            MeshFilter filter = prefab.GetComponentInChildren<MeshFilter>();
            if (filter != null && filter.sharedMesh != null)
            {
                Debug.LogWarning($"No Mesh: {name}");
                cloth.clothBlueprint = GenerateBlueprints(filter.sharedMesh);
                AssetDatabase.CreateAsset(cloth.clothBlueprint, blueprintPath);
                PrefabUtility.SaveAsPrefabAsset(solver.gameObject, prefabPath);
            }
            DestroyImmediate(prefab);
        }
        AssetDatabase.Refresh();
        DestroyImmediate(solver.gameObject);
    }



    static ObiClothBlueprint GenerateBlueprints(Mesh mesh)
    {
        float mass = 0.0005f;
        float radius = 0.01f;
        ObiClothBlueprint blueprint = ScriptableObject.CreateInstance<ObiClothBlueprint>();
        blueprint.inputMesh = mesh;
        blueprint.GenerateImmediate();
        for (int i = 0; i < blueprint.invMasses.Length; i++)
        {
            blueprint.invMasses[i] = 1f / mass;
        }
        for (int i = 0; i < blueprint.principalRadii.Length; i++)
        {
            float maxRad = Mathf.Max(blueprint.principalRadii[i].x, blueprint.principalRadii[i].y, blueprint.principalRadii[i].z);
            float mul = radius / maxRad;
            blueprint.principalRadii[i] = blueprint.principalRadii[i] * mul;
        }
        return blueprint;
    }
}
