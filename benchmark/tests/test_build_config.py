from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET


PROJECT_FILE = Path(__file__).resolve().parents[2] / "src" / "cnumpy_ahk.vcxproj"
MSBUILD_NAMESPACE = "http://schemas.microsoft.com/developer/msbuild/2003"
NS = {"msbuild": MSBUILD_NAMESPACE}
RELEASE_X64_CONDITION = re.compile(
    r"^\s*'\$\(Configuration\)\|\$\(Platform\)'\s*==\s*'Release\|x64'\s*$",
    re.IGNORECASE,
)


def release_x64_compiler_settings(project):
    settings = {}
    for group in project.findall("msbuild:ItemDefinitionGroup", NS):
        condition = group.attrib.get("Condition", "")
        if RELEASE_X64_CONDITION.fullmatch(condition) is None:
            continue

        compiler = group.find("msbuild:ClCompile", NS)
        if compiler is None:
            continue
        for setting in compiler:
            setting_name = setting.tag.split("}", 1)[-1]
            settings[setting_name] = setting.text
    return settings


class ReleaseCompilerSettingsParserTests(unittest.TestCase):
    def test_allows_expression_whitespace_and_applies_later_overrides(self):
        project = ET.fromstring(
            f"""
            <Project xmlns="{MSBUILD_NAMESPACE}">
              <ItemDefinitionGroup
                  Condition=" '$(Configuration)|$(Platform)'  ==  'Release|x64' ">
                <ClCompile>
                  <Optimization>Disabled</Optimization>
                  <FloatingPointModel>Precise</FloatingPointModel>
                </ClCompile>
              </ItemDefinitionGroup>
              <ItemDefinitionGroup
                  Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
                <ClCompile>
                  <Optimization>MaxSpeed</Optimization>
                  <EnableEnhancedInstructionSet>StreamingSIMDExtensions2</EnableEnhancedInstructionSet>
                </ClCompile>
              </ItemDefinitionGroup>
            </Project>
            """
        )

        self.assertEqual(
            {
                "Optimization": "MaxSpeed",
                "FloatingPointModel": "Precise",
                "EnableEnhancedInstructionSet": "StreamingSIMDExtensions2",
            },
            release_x64_compiler_settings(project),
        )

    def test_rejects_whitespace_inside_quoted_operands(self):
        project = ET.fromstring(
            f"""
            <Project xmlns="{MSBUILD_NAMESPACE}">
              <ItemDefinitionGroup
                  Condition="'$(Configuration)|$(Platform)' == 'Release | x64'">
                <ClCompile>
                  <Optimization>MaxSpeed</Optimization>
                </ClCompile>
              </ItemDefinitionGroup>
            </Project>
            """
        )

        self.assertEqual({}, release_x64_compiler_settings(project))

    def test_matches_condition_operands_case_insensitively(self):
        project = ET.fromstring(
            f"""
            <Project xmlns="{MSBUILD_NAMESPACE}">
              <ItemDefinitionGroup
                  Condition="'$(Configuration)|$(Platform)'=='Release|x64'">
                <ClCompile>
                  <Optimization>Disabled</Optimization>
                </ClCompile>
              </ItemDefinitionGroup>
              <ItemDefinitionGroup
                  Condition="'$(Configuration)|$(Platform)' == 'release|X64'">
                <ClCompile>
                  <Optimization>MaxSpeed</Optimization>
                </ClCompile>
              </ItemDefinitionGroup>
            </Project>
            """
        )

        self.assertEqual(
            {"Optimization": "MaxSpeed"},
            release_x64_compiler_settings(project),
        )


class ReleaseBuildConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project = ET.parse(PROJECT_FILE).getroot()

    def test_release_x64_uses_required_optimization_settings(self):
        expected = {
            "Optimization": "MaxSpeed",
            "FloatingPointModel": "Fast",
            "EnableEnhancedInstructionSet": "StreamingSIMDExtensions2",
        }
        settings = release_x64_compiler_settings(self.project)
        actual = {
            setting: settings.get(setting)
            for setting in expected
        }
        self.assertEqual(expected, actual)

    def test_project_compiles_simd_implementation(self):
        sources = {
            item.attrib["Include"].replace("\\", "/").casefold()
            for item in self.project.findall(".//msbuild:ClCompile", NS)
            if "Include" in item.attrib
        }
        self.assertIn("simd_ops.c", sources)

    def test_avx2_file_has_an_isolated_non_ltcg_compiler_boundary(self):
        items = {
            item.attrib.get("Include", "").replace("\\", "/").casefold(): item
            for item in self.project.findall(".//msbuild:ClCompile", NS)
        }
        self.assertIn("simd_dispatch.c", items)
        self.assertIn("simd_avx2.c", items)
        avx2 = items["simd_avx2.c"]
        settings = {
            child.tag.split("}", 1)[-1]: child.text
            for child in avx2
        }
        self.assertEqual(
            "AdvancedVectorExtensions2",
            settings["EnableEnhancedInstructionSet"],
        )
        self.assertEqual("false", settings["WholeProgramOptimization"])

    def test_gemm_avx2_has_an_isolated_non_ltcg_compiler_boundary(self):
        items = {
            item.attrib.get("Include", "").replace("\\", "/").casefold(): item
            for item in self.project.findall(".//msbuild:ClCompile", NS)
        }
        for source_name in ("gemm.c", "gemm_sse2.c", "gemm_avx2.c"):
            with self.subTest(source=source_name):
                self.assertIn(source_name, items)
        avx2 = items["gemm_avx2.c"]
        settings = {
            child.tag.split("}", 1)[-1]: child.text
            for child in avx2
        }
        self.assertEqual(
            "AdvancedVectorExtensions2",
            settings["EnableEnhancedInstructionSet"],
        )
        self.assertEqual("false", settings["WholeProgramOptimization"])
        for baseline_name in ("gemm.c", "gemm_sse2.c"):
            baseline_settings = {
                child.tag.split("}", 1)[-1]: child.text
                for child in items[baseline_name]
            }
            self.assertNotEqual(
                "AdvancedVectorExtensions2",
                baseline_settings.get("EnableEnhancedInstructionSet"),
            )


if __name__ == "__main__":
    unittest.main()
