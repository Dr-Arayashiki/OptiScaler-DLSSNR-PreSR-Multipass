#include "pch.h"
#include "DlssNrFeature_Vk.h"

#include "DlssNr.h"
#include "DlssNr_ExposureScan.h"

#include <Config.h>
#include <menu/menu_common.h>

#include <imgui/imgui.h>

#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace DlssNr
{

// The "(?)" marker every control carries, matching the rest of the menu.
static void HelpMarker(const char* tip)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");

    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 40.0f);
        ImGui::TextUnformatted(tip);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

// A slider that only writes its value when the handle is released.
//
// Some controls -- intensity, the structure and tone strengths -- are read by the model once, when
// the feature is built, so changing one rebuilds the whole feature. Writing on every pixel of a drag
// meant a rebuild per frame, felt as the picture hitching while you scrub. The slider still tracks
// live under the cursor; only the commit that triggers the rebuild waits for release. Cheap controls
// that are just shader constants (detail, colour, paper white) do not use this -- they can afford to
// apply live.
template <typename Option>
static bool DeferredSlider(const char* label, Option* opt, float mn, float mx, float def, const char* fmt = "%.2f",
                           bool inheritReset = false, const char* resetLabel = nullptr)
{
    static std::unordered_map<ImGuiID, float> pending;
    const ImGuiID id = ImGui::GetID(label);

    auto it = pending.find(id);
    float value = it != pending.end() ? it->second : (opt->has_value() ? opt->value() : def);
    bool changed = false;

    if (ImGui::SliderFloat(label, &value, mn, mx, fmt))
        pending[id] = value;

    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        auto committed = pending.find(id);

        if (committed != pending.end())
        {
            *opt = std::clamp(committed->second, mn, mx);
            pending.erase(committed);
            changed = true;
        }
    }

    ImGui::SameLine();

    const std::string resetId = std::string(resetLabel ? resetLabel : "Reset") + "##" + label;
    if (ImGui::SmallButton(resetId.c_str()))
    {
        if (inheritReset)
            *opt = std::optional<float> {};
        else
            *opt = def;
        pending.erase(id);
        changed = true;
    }

    return changed;
}

// An absent later-pass setting inherits pass 1. The first combo item represents that absence; the
// remaining items map directly to the model's zero-based profile values.
static bool InheritedProfileCombo(const char* label, CustomOptional<uint32_t, NoDefault>* opt, const char* const* names,
                                  int nameCount)
{
    int selected = 0;

    if (opt->has_value())
        selected = std::clamp((int) opt->value(), 0, nameCount - 2) + 1;

    if (!ImGui::Combo(label, &selected, names, nameCount))
        return false;

    if (selected == 0)
        *opt = std::optional<uint32_t> {};
    else
        *opt = (uint32_t) (selected - 1);

    return true;
}

void RenderMenu(Config* config, float menuResScale)
{

    // DLSS Neural Rendering -----------------------------
    ImGui::Spacing();
    if (auto ch = ScopedCollapsingHeader("DLSS Neural Rendering"); ch.IsHeaderOpen())
    {
        ScopedIndent indent {};
        ImGui::Spacing();

        bool ptBr = config->DlssNrTranslatePtBr.value_or_default();
        if (ImGui::Checkbox(ptBr ? "Traduzir aba para Português do Brasil [PT-BR]##ptbr"
                                 : "Translate tab to Brazilian Portuguese [PT-BR]##ptbr",
                            &ptBr))
            config->DlssNrTranslatePtBr = ptBr;
        ImGui::Spacing();

        auto T = [ptBr](const char* en, const char* pt) -> const char* { return ptBr ? pt : en; };
        const char* resetText = ptBr ? "Restaurar" : "Reset";

        bool enabled = config->DlssNrEnabled.value_or_default();
        if (ImGui::Checkbox(T("Enable Neural Rendering", "Ativar Neural Rendering (DLSS 5)"), &enabled))
            config->DlssNrEnabled = enabled;

        HelpMarker(T("Synthesises detail in the upscaler's frame, before frame generation sees it."
                     "\n\nNeeds two similarly named files beside OptiScaler, one character apart:"
                     "\n  nvngx_dlssnr.dll       NVIDIA's model (~165 MB) -- you supply it"
                     "\n  nvngx.dll_dlssnr.dll   the forwarder (~13 KB) -- ships in this package"
                     "\nUndocumented and driven directly, so none of this is officially supported.",
                     "Sintetiza detalhes no quadro do upscaler, antes da geração de quadros (FG)."
                     "\n\nRequer dois arquivos ao lado do OptiScaler:"
                     "\n  nvngx_dlssnr.dll       Modelo da NVIDIA (~165 MB) -- fornecido por você"
                     "\n  nvngx.dll_dlssnr.dll   Forwarder (~13 KB) -- incluso neste pacote"
                     "\nRecurso de renderização neural controlado diretamente via NGX."));

        bool beforeSr = config->DlssNrRunBeforeSr.value_or_default();
        const bool deferredActive = config->DlssNrDeferredDlss.value_or_default();
        if (deferredActive)
            ImGui::BeginDisabled();
        if (ImGui::Checkbox(T("Apply before Super Resolution", "Aplicar antes do Super Resolution (Pre-SR)"),
                            &beforeSr))
            config->DlssNrRunBeforeSr = beforeSr;
        if (deferredActive)
            ImGui::EndDisabled();

        HelpMarker(T("Runs Neural Rendering on the render-resolution colour input immediately before"
                     "\nSuper Resolution, so SR temporally accumulates and upscales the enhanced frame."
                     "\n\nRay Reconstruction is deliberately excluded: its input contract differs and"
                     "\nuses the separate Apply after Ray Reconstruction option. Origin-zero padded"
                     "\ninputs use their active render size; offset or invalid inputs fall back post-SR."
                     "\n\nThis placement control currently applies to the Direct3D 12 path and its"
                     "\nDirect3D 11/Vulkan bridges; native Vulkan keeps the post-upscale path.",
                     "Executa o Neural Rendering na resolução de renderização imediatamente antes"
                     "\ndo Super Resolution, acumulando temporalmente os detalhes aprimorados."
                     "\n\nRay Reconstruction possui contrato separado através da opção de RR."
                     "\nCompatível com Direct3D 12 e bridges D3D11/Vulkan."));

        bool deferredDlss = config->DlssNrDeferredDlss.value_or_default();
        if (ImGui::Checkbox(
                T("Generate before SR, apply after SR (DLSS)", "Gerar antes do SR, aplicar após SR (DLSS Residual)"),
                &deferredDlss))
            config->DlssNrDeferredDlss = deferredDlss;
        HelpMarker(T("Experimental: the game raster stays clean. NR runs on a render-size copy, then"
                     "\na private DLSS SR feature upscales its signed contribution for application after SR."
                     "\nThe contribution is encoded around neutral grey; DLSS may distort it or flicker."
                     "\nRequires an NVIDIA DLSS SR runtime. No spatial/FSR fallback is used on failure."
                     "\nOverrides the pre-SR checkbox, not RR. Native Vulkan is not supported."
                     "\nNR's displayed GPU time excludes the additional DLSS/composition cost."
                     "\nDisable frame hold, debug views and comparison modes for this experiment.",
                     "Experimental: a rasterização do jogo fica limpa. O NR roda em uma cópia na"
                     "\nresolução de render e um feature DLSS SR privado escala a contribuição pós-SR."
                     "\nRequer runtime DLSS SR da NVIDIA. Sobrepõe o Pre-SR."));
        if (deferredDlss)
            ImGui::TextWrapped(T("Residual DLSS: %s", "DLSS Residual: %s"), DlssNr::DeferredDlssStatus().c_str());
        ImGui::BeginDisabled(!deferredDlss);
        bool asyncLatest = config->DlssNrAsyncLatest.value_or_default();
        if (ImGui::Checkbox(T("Async NR: use latest completed result (experimental)",
                              "NR Assíncrono: usar resultado mais recente concluído (experimental)"),
                            &asyncLatest))
            config->DlssNrAsyncLatest = asyncLatest;
        HelpMarker(T("Runs NR and private residual DLSS on a separate GPU queue. The CURRENT raster"
                     "\nuses the newest completed result, even if it is several frames old."
                     "\nNo raster delay, residual interpolation or GPU wait for NR completion."
                     "\nBusy requests are dropped; cuts/resizes discard stale results."
                     "\nRegular FG keeps the current raster and its current guides."
                     "\nGPU contention, snapshot copies, CPU API submission and model rebuilds still"
                     "\nhave a cost: this is not a guarantee of unaffected FPS or frame pacing."
                     "\nOverrides the two-frame residual FG/hold controls below. Histories reset per NR job.",
                     "Executa o NR e DLSS residual em fila assíncrona separada na GPU."
                     "\nO quadro atual usa o resultado mais recente concluído sem travar a renderização."));
        ImGui::BeginDisabled(asyncLatest);
        bool residualFg = config->DlssNrResidualFg.value_or_default();
        if (ImGui::Checkbox(T("NR every second frame (NVIDIA FG, experimental)",
                              "NR a cada dois quadros (NVIDIA FG, experimental)"),
                            &residualFg))
            config->DlssNrResidualFg = residualFg;
        HelpMarker(T("Uses NVIDIA interpolation on the DLSS-upscaled residual, not full game frames."
                     "\nBuffers the matching clean SR image and residual by one rendered frame."
                     "\nWARNING: later game effects may still use newer-frame data; motion blur,"
                     "\nlighting effects and UI may not align. Adds latency and GPU/VRAM cost."
                     "\nRequires a working NVIDIA FG runtime, low-resolution non-jittered motion vectors."
                     "\nWith no motion texture: sample-and-hold applies each residual to two CURRENT"
                     "\nframes, with no residual FG or raster delay. Fresh private NR/SR samples use"
                     "\nzero guides with history reset; the game's own upscaler still needs valid inputs."
                     "\nCuts and failed evaluations reset history; rejected FG output retains clean colour.",
                     "Utiliza interpolação NVIDIA FG no residual do DLSS em vez do quadro completo."
                     "\nRequer runtime NVIDIA FG e vetores de movimento válidos."));
        bool approxCamera = config->DlssNrResidualFgApproxCamera.value_or_default();
        if (ImGui::Checkbox(T("Allow approximate FG camera guides (experimental)",
                              "Permitir guias de câmera aproximadas para FG (experimental)"),
                            &approxCamera))
            config->DlssNrResidualFgApproxCamera = approxCamera;
        HelpMarker(T("Explicit fallback for games/bridges without full camera transforms."
                     "\nUses real motion vectors, but approximate projection and camera transforms."
                     "\nCamera movement and disocclusions may produce artifacts. Not reference-quality guides.",
                     "Fallback para jogos/bridges sem matrizes completas de câmera."));
        ImGui::EndDisabled();
        ImGui::EndDisabled();

        bool afterRR = config->DlssNrApplyAfterRR.value_or_default();
        if (ImGui::Checkbox(T("Apply after Ray Reconstruction (DX12)", "Aplicar após Ray Reconstruction (DX12)"),
                            &afterRR))
            config->DlssNrApplyAfterRR = afterRR;
        HelpMarker(T("Requires the game's native Ray Reconstruction option. RR denoises and upscales"
                     "\nfirst; NR then processes its output before frame generation."
                     "\nThis does not add RR to games without the required rendering buffers."
                     "\nIndependent controls below prevent inheriting the cost of the SR configuration."
                     "\nNative Vulkan does not use these DX12 controls.",
                     "Requer Ray Reconstruction nativo no jogo. O RR desrui e escala primeiro,"
                     "\ne o NR processa o resultado antes do Frame Generation."));
        int rrPasses = (int) config->DlssNrRRPasses.value_or_default();
        if (ImGui::SliderInt(
                T("NR passes after RR", "Passes de NR após RR"), &rrPasses, 1,
                (int) (config->DlssNrUnlockPasses.value_or_default() ? MaxPassCount : DefaultMaxPassCount)))
            config->DlssNrRRPasses = (unsigned int) rrPasses;
        float rrScale = config->DlssNrRRWorkingScale.value_or_default();
        if (ImGui::SliderFloat(T("NR model scale after RR", "Escala do modelo NR após RR"), &rrScale, 0.25f, 2.0f,
                               "%.2fx"))
            config->DlssNrRRWorkingScale = rrScale;
        HelpMarker(T("Relative to RR's OUTPUT resolution: 0.50x at 4K runs NR at 1920x1080."
                     "\nRR itself remains full quality. The NR edit is resized for final composition."
                     "\nPasses 2 and 3 use the same per-pass model profiles as the SR path.",
                     "Relativo à resolução de saída do RR: 0.50x em 4K executa o NR a 1080p."
                     "\nO RR permanece em qualidade total e o efeito neural é redimensionado."));

        // The toggle can be bound to a key, and nobody would think to look for it under Keybinds
        // unless told. Dimmed, because it is a note rather than a setting.
        ImGui::TextDisabled("%s",
                            T("Can be toggled with a key -- bind it under Keybinds, \"Neural Rendering\".",
                              "Pode ser alternado por tecla -- configure em Teclas de Atalho, \"Neural Rendering\"."));

        bool applyModel = config->DlssNrApplyModel.value_or_default();
        if (ImGui::Checkbox(T("Apply the model", "Aplicar o modelo neural"), &applyModel))
            config->DlssNrApplyModel = applyModel;

        HelpMarker(T("Whether the model's edit is applied. Off shows the clean upscaler frame while the"
                     "\npass keeps running -- so with Hold frame (under Compare) you can freeze a"
                     "\nframe and toggle this to see the same frozen frame with and without Neural"
                     "\nRendering. Leave it on for normal use.",
                     "Define se o resultado do modelo é aplicado à tela. Se desmarcado, exibe o quadro"
                     "\nlimpo enquanto o passe continua rodando (ideal para comparar A/B com quadro congelado)."));

        const bool vulkan = DlssNr::IsRunningVk();

        if (!enabled)
        {
            ImGui::TextDisabled("%s", T("Off. The model stays loaded, so turning this back on is immediate.",
                                        "Desativado. O modelo permanece em memória, reativação imediata."));
        }
        else if (!DlssNr::IsRunning() && !vulkan)
        {
            const auto feature = State::Instance().currentFeature;
            const bool nativeVk = feature && feature->Api() == API::Vulkan && !feature->IsWithDx12();
            const char* reason = nativeVk ? DlssNr::FailureReasonVk() : DlssNr::FailureReason();

            if (reason[0] != 0)
            {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.35f, 1.0f),
                                   T("Off for this session: %s.", "Desativado nesta sessão: %s."), reason);
                ImGui::SameLine();

                if (nativeVk)
                    ImGui::TextUnformatted(T("Restart the game to retry native Vulkan NR.",
                                             "Reinicie o jogo para tentar novamente no Vulkan nativo."));
                else if (ImGui::SmallButton(T("Retry", "Tentar novamente")))
                    DlssNr::RetryAfterFailure();
            }
            else if (feature && feature->Api() == API::DX11 && !feature->IsWithDx12())
            {
                ImGui::TextWrapped(
                    "%s", T("NR needs the D3D12 bridge on D3D11. Choose an upscaler marked w/Dx12 and restart.",
                            "O NR requer bridge D3D12 no D3D11. Selecione um upscaler com sufixo w/Dx12 e reinicie."));
            }
            else if (nativeVk && config->DlssNrDeferredDlss.value_or_default())
            {
                ImGui::TextWrapped(
                    "%s", T("Deferred DLSS residual composition and async NR currently require D3D12 or its bridge."
                            " Disable Generate before SR, apply after SR (DLSS) to use native Vulkan NR."
                            " Apply before Super Resolution is supported here.",
                            "DLSS Residual diferido e NR assíncrono requerem D3D12 ou seu bridge."));
            }
            else if (enabled)
                ImGui::TextUnformatted(T("Waiting for the upscaler to run.", "Aguardando o upscaler executar."));
        }
        else
        {
            const auto ms = vulkan ? DlssNr::LastGpuTimeVk() : DlssNr::LastGpuTime();

            const char* runSuffix = !config->DlssNrApplyModel.value_or_default()
                                        ? T("  (model running, edit hidden)", "  (modelo executando, edição oculta)")
                                        : "";

            if (ms.has_value())
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f),
                                   T("Running%s - %.2f ms per frame%s", "Em execução%s - %.2f ms por quadro%s"),
                                   vulkan ? T(" natively on Vulkan", " nativamente no Vulkan") : "", ms.value(),
                                   runSuffix);
            else if (vulkan)
                ImGui::TextColored(
                    ImVec4(0.4f, 0.9f, 0.5f, 1.0f),
                    T("Running natively on Vulkan - %llu frames%s", "Em execução nativa no Vulkan - %llu quadros%s"),
                    DlssNr::FramesVk(), runSuffix);
            else
                ImGui::TextColored(ImVec4(0.4f, 0.9f, 0.5f, 1.0f), T("Running.%s", "Em execução.%s"), runSuffix);

            ImGui::SameLine();
            ImGui::TextDisabled("(?)");
            if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
                ImGui::SetTooltip("%s", T("The whole pass: the staging copies and the resolve as well as the"
                                          "\nmodel. Timing only the model would flatter the number."
                                          "\n\nCompare it against the frame time at the bottom of this window to"
                                          "\nsee what it is costing you.",
                                          "Tempo total do passe: cópias de staging, inferência do modelo e composição."
                                          "\nCompare com o tempo de quadro total para avaliar o impacto."));
        }

        ImGui::Spacing();
        ImGui::PushItemWidth(220.0f * menuResScale);

        ImGui::SeparatorText(T("Cost", "Custo de Desempenho"));

        bool unlockPasses = config->DlssNrUnlockPasses.value_or_default();
        if (ImGui::Checkbox(
                T("Lift model pass limit (up to 30; expensive)", "Remover limite de passes (até 30; alto custo)"),
                &unlockPasses))
            config->DlssNrUnlockPasses = unlockPasses;
        HelpMarker(T("Optional extended pass range, inspired by y4my4my4m's fork."
                     "\nThe normal ceiling remains 3. More passes cost GPU time and VRAM and can amplify artifacts."
                     "\nHigh counts can exhaust VRAM or trigger a driver timeout. Raise gradually."
                     "\nNo extra model features are allocated unless those passes are active.",
                     "Faixa estendida de passes. O padrão é até 3 passes."
                     "\nMais passes aumentam o uso de GPU e VRAM e podem amplificar artefatos."));
        const unsigned int passLimit = unlockPasses ? MaxPassCount : DefaultMaxPassCount;

        {
            int passes = (int) std::clamp(config->DlssNrPasses.value_or_default(), 1u, passLimit);
            const ImVec4 colour = passes <= 1   ? ImVec4(0.35f, 0.88f, 0.38f, 1.0f)
                                  : passes == 2 ? ImVec4(0.95f, 0.70f, 0.20f, 1.0f)
                                                : ImVec4(0.92f, 0.30f, 0.25f, 1.0f);

            ImGui::PushStyleColor(ImGuiCol_Text, colour);
            ImGui::PushStyleColor(ImGuiCol_SliderGrab, colour);

            const char* fmt1 =
                passes == 1 ? T("%d (normal)", "%d (normal)") : T("%dx model cost", "%dx custo do modelo");
            if (ImGui::SliderInt(T("Model passes", "Passes do modelo"), &passes, 1, (int) passLimit, fmt1))
                config->DlssNrPasses = (uint32_t) std::clamp(passes, 1, (int) passLimit);

            ImGui::PopStyleColor(2);

            HelpMarker(
                T("Runs sequential model layers between one encode and one final composition."
                  "\nEach additional layer consumes the previous layer's model output and owns"
                  "\na separate persistent feature and temporal history."
                  "\n\nThe base proxy stays immutable and the final answer is composed against it"
                  "\nonce, so colour and transfer strength do not compound. Later layers default"
                  "\nto zero local tone; each layer can override it."
                  "\n\nCost scales almost linearly. Two is the common 'deep fried' look; three is"
                  "\nthe guarded ceiling because later layers converge while cost and artifacts grow.",
                  "Executa camadas sequenciais do modelo neural entre a codificação e a composição final."
                  "\nCada camada adicional consome a saída da anterior com histórico temporal independente."
                  "\nO custo escala quase linearmente. 2 passes gera maior definição; 3 é o limite recomendado."));
        }

        static int pendingScale = -1;

        int scalePercent =
            pendingScale >= 0 ? pendingScale : (int) lroundf(config->DlssNrWorkingScale.value_or_default() * 100.0f);

        if (ImGui::SliderInt(T("Model resolution", "Resolução do modelo"), &scalePercent, 25, 200, "%d%%"))
            pendingScale = scalePercent;

        if (ImGui::IsItemDeactivatedAfterEdit() && pendingScale >= 0)
        {
            config->DlssNrWorkingScale = std::clamp(pendingScale, 25, 200) / 100.0f;
            pendingScale = -1;
        }

        if (scalePercent > 100)
            ImGui::TextDisabled(
                "%s", T(std::format("Supersampling {:.2f}x: the model runs ABOVE native, then\n"
                                    "is sampled back down. Experimental, and costly -- time grows with the area.",
                                    scalePercent / 100.0f)
                            .c_str(),
                        std::format("Supersampling {:.2f}x: o modelo executa ACIMA da resolução nativa,\n"
                                    "e é reduzido depois. Experimental e custoso.",
                                    scalePercent / 100.0f)
                            .c_str()));

        if (scalePercent > 100)
        {
            static const char* dsNames[] = { "FSR1",     "Bicubic", "Catmull-Rom", "Lanczos2",
                                             "Lanczos3", "Kaiser2", "Kaiser3",     "MAGIC" };
            int ds = (int) config->DlssNrScalingDownscaler.value_or_default();
            if (ds < 0 || ds >= IM_ARRAYSIZE(dsNames))
                ds = (int) Scaler::Lanczos3;

            if (ImGui::Combo(T("Downscaler (NR)", "Redutor / Downscaler (NR)"), &ds, dsNames, IM_ARRAYSIZE(dsNames)))
                config->DlssNrScalingDownscaler = (Scaler) ds;

            HelpMarker(T("The filter that averages the model's above-native answer back to display size --"
                         "\nthis is what turns supersampling into LESS noise rather than more. Sharper"
                         "\nfilters (Lanczos3, Kaiser3) keep the most detail; softer ones (Bicubic,"
                         "\nCatmull-Rom) are gentler on ringing. Independent of the Output Scaling"
                         "\ndownscaler, so the two can differ and run at the same time.",
                         "Filtro que reduz a saída supersample do modelo para o tamanho de exibição."
                         "\nFiltros mais nítidos (Lanczos3, Kaiser3) preservam mais detalhes; filtros mais"
                         "\nsuaves (Bicubic, Catmull-Rom) evitam ringing."));
        }

        HelpMarker(T("What fraction of the frame the model works at. Cost falls with the square of"
                     "\nthis, so half resolution is roughly a quarter of the time."
                     "\n\nThe frame is never reduced. Only the model's contribution is computed small"
                     "\nand enlarged, so the picture underneath is untouched whatever this says."
                     "\n\nWhat it trades: the shading the model adds is broad and survives enlargement;"
                     "\nthe fine structure it synthesises does not, and softens. Worth having when the"
                     "\npass costs more than you want to pay for the detail it returns."
                     "\n\nThe frame itself stays at full detail whatever this says -- only the"
                     "\nmodel's own work is done small.",
                     "Fração da resolução em que o modelo trabalha. O custo cai quadraticamente."
                     "\nO quadro do jogo nunca é reduzido, apenas o processamento neural."));

        {
            const bool reduced = config->DlssNrWorkingScale.value_or_default() < 0.999f;

            if (!reduced)
                ImGui::BeginDisabled();

            static const char* enlargeNamesEn[] = { "Classic", "Matched residual" };
            static const char* enlargeNamesPt[] = { "Clássico", "Residual correspondente" };
            int enlarge = config->DlssNrTransfer.value_or_default() == 1 ? 1 : 0;

            if (ImGui::Combo(T("Enlargement", "Ampliação"), &enlarge, ptBr ? enlargeNamesPt : enlargeNamesEn, 2))
                config->DlssNrTransfer = (uint32_t) enlarge;

            if (!reduced)
                ImGui::EndDisabled();

            HelpMarker(T("How the model's work is brought back up when it ran below the frame's size."
                         "\n\nClassic composes the model's small picture directly against the full-size"
                         "\nframe. Those two disagree by the shrink's blur as well as by the model's edit,"
                         "\nand the composition cannot tell them apart -- it reads the blur as brightness"
                         "\nthe frame has and the model never saw. The lower the model resolution the"
                         "\nlarger that error, and it is the colour shift that shows up at 50%."
                         "\n\nMatched residual carries up only the model's difference and lays it on the"
                         "\nframe's own proxy, so both pictures being compared are full size and the only"
                         "\nthing that came from the small raster is the edit itself."
                         "\n\nNo effect at 100% or above: there is no residual to carry and the two are"
                         "\nidentical (supersampling brings its answer down to frame size before this)."
                         "\n\nFrom hhkbble's multi-pass work on this fork.",
                         "Método de reconstrução da saída quando o modelo roda abaixo da resolução nativa."
                         "\nResidual correspondente preserva fidelidade de cores e nitidez."));
        }

        ImGui::SeparatorText(T("How much of it lands", "Intensidade e Mesclagem"));

        float transfer = config->DlssNrTransferStrength.value_or_default();
        if (ImGui::SliderFloat(T("Detail strength", "Força de detalhes"), &transfer, 0.0f, 2.0f, "%.2f"))
            config->DlssNrTransferStrength = transfer;

        ImGui::SameLine();
        if (ImGui::SmallButton(T("Reset##detail", "Restaurar##detail")))
            config->DlssNrTransferStrength = 1.0f;

        HelpMarker(T("How far the frame moves toward the model's picture."
                     "\n\nThe model's answer is not added to the frame -- it is a complete picture of its"
                     "\nown, rescaled so its luminance sits where the original says it should. This"
                     "\nblends between the two, so both ends are real pictures and everything between"
                     "\nthem is one too."
                     "\n\n0 gives back exactly what the upscaler produced. 1 is the model's picture."
                     "\n\nAbove 1 carries on past it in the same direction, which is not something the"
                     "\nmodel asked for -- use it to see what it is doing, then come back down. This"
                     "\nis the control to push if you want more effect: Intensity belongs to the model"
                     "\nand it decides what to do with it.",
                     "Quanto da imagem final se aproxima do resultado gerado pela rede neural."
                     "\n0 mantém o upscaler puro. 1 aplica o modelo completo. Acima de 1 acentua os detalhes."));

        float colour = config->DlssNrColourStrength.value_or_default();
        if (ImGui::SliderFloat(T("Colour strength", "Força de cores"), &colour, 0.0f, 4.0f, "%.2f"))
            config->DlssNrColourStrength = colour;

        ImGui::SameLine();
        if (ImGui::SmallButton(T("Reset##colour", "Restaurar##colour")))
            config->DlssNrColourStrength = 1.0f;

        HelpMarker(T("Whether the model's colour arrives with its light."
                     "\n\n0 keeps the game's own hue exactly -- every pixel is the original colour with"
                     "\nonly its brightness carrying the model's verdict. Game-accurate colour, with"
                     "\nthe detail. 1 brings the model's colour as well, in its own hue, clamped into"
                     "\nAP1 so nothing unreachable is asked for."
                     "\n\nThis cannot shift hue on its own: it interpolates between two finished"
                     "\npictures rather than adding a colour difference to one, which is what used to"
                     "\nlet a warm subject come back green."
                     "\n\nAbove 1 it OVER-SATURATES: the colour keeps its hue but grows more vivid,"
                     "\nand rolls off at the edge of what the display can show rather than clipping"
                     "\ninto a flat blown patch. 1 is the model's own colour; push past it for punch.",
                     "Controla se as cores geradas pelo modelo são incorporadas ao quadro."
                     "\n0 preserva estritamente as tonalidades do jogo; 1 incorpora a coloração neural."));

        static const char* reversibleNamesEn[] = { "Off (soft knee)", "Neutwo proxy + composed",
                                                   "Neutwo proxy + replace", "Hybrid proxy + composed",
                                                   "Hybrid proxy + replace" };
        static const char* reversibleNamesPt[] = { "Desativado (soft knee)", "Proxy Neutwo + composto",
                                                   "Proxy Neutwo + substituição", "Proxy Híbrido + composto",
                                                   "Proxy Híbrido + substituição" };
        int reversible = (int) config->DlssNrReversibleMode.value_or_default();
        if (reversible < 0 || reversible > 4)
            reversible = 0;
        if (ImGui::Combo(T("Reversible proxy (experimental)", "Proxy reversível (experimental)"), &reversible,
                         ptBr ? reversibleNamesPt : reversibleNamesEn, 5))
            config->DlssNrReversibleMode = (uint32_t) reversible;

        HelpMarker(T("What the model is shown, and how its answer comes back."
                     "\n\nOff (soft knee): the default. It rolls highlights off so hard the model"
                     "\ncannot resolve detail in them -- fine in soft-lit scenes, weak in bright ones."
                     "\n\nNeutwo composed: an unclipped curve so the model sees highlight detail, then"
                     "\neverything above (Detail/Colour strength, highlight guard, palette). It wins in"
                     "\nbright scenes, but the curve compresses MIDTONES too, so in soft-lit content it"
                     "\ncan be worse than Off. It also shifts paper white -- re-check it when you switch."
                     "\n\nHybrid composed: the best of both, and the one to use. Identity in the"
                     "\nmidtones -- as good as Off there -- and the unclipped roll only in the"
                     "\nhighlights, so it recovers the detail Off crushes without giving up the"
                     "\nmidtones Neutwo does. It barely shifts paper white."
                     "\n\nReplace: the raw model straight back through the exact inverse, none of the"
                     "\ncomposition -- no guard, no palette, no strengths. Gorgeous where there are no"
                     "\nbright lights, but they FLASH in motion. A reference, not a daily setting."
                     "\n\nHybrid replace: the raw model like Replace, but on the hybrid curve -- the"
                     "\ndecode is identity in the midtones, so the flashing is confined to genuine"
                     "\nbright highlights instead of everywhere. Most of Replace's detail, far more"
                     "\nstable. If you love the Replace look but the flicker bothers you, use this."
                     "\n\nOff is byte-identical to before.",
                     "Como a imagem HDR é codificada para a rede neural e decodificada na volta."
                     "\nHybrid Composed oferece o melhor equilíbrio para realces e médios tons sem cintilação."));

        ImGui::SeparatorText(T("Model passes", "Passes do Modelo Neural"));
        ImGui::TextWrapped(
            "%s", T("Each pass has its own style and model strengths. Changes apply when you release a slider.",
                    "Cada passe possui seu estilo e forças neurais. As alterações são aplicadas ao soltar o controle "
                    "deslizante."));
        static const char* stylesEn[] = { "Standard", "Natural", "Cinematic" };
        static const char* stylesPt[] = { "Padrão", "Natural", "Cinematográfico" };
        static const char* inheritedStylesEn[] = { "Auto (inherit pass 1)", "Standard", "Natural", "Cinematic" };
        static const char* inheritedStylesPt[] = { "Auto (herdar do passe 1)", "Padrão", "Natural", "Cinematográfico" };

        if (ImGui::TreeNodeEx(T("Pass 1", "Passe 1"), ImGuiTreeNodeFlags_DefaultOpen))
        {
            int style = (int) std::min(config->DlssNrStyle.value_or_default(), 2u);
            if (ImGui::Combo(T("Style", "Estilo"), &style, ptBr ? stylesPt : stylesEn, 3))
                config->DlssNrStyle = (uint32_t) style;
            DeferredSlider(T("Intensity", "Intensidade"), &config->DlssNrIntensity, 0.0f, 2.0f, 1.0f, "%.2f", false,
                           resetText);
            DeferredSlider(T("Local structure", "Estrutura local"), &config->DlssNrLocalStructure, 0.0f, 2.0f, 1.0f,
                           "%.2f", false, resetText);
            DeferredSlider(T("Local tone", "Tom local"), &config->DlssNrLocalTone, 0.0f, 2.0f, 1.0f, "%.2f", false,
                           resetText);
            DeferredSlider(T("Skin structure", "Estrutura de pele"), &config->DlssNrSkinStructure, -1.0f, 2.0f, -1.0f,
                           "%.2f", false, resetText);
            HelpMarker(
                T("-1 follows local structure; 0 reduces skin structure independently."
                  "\nThis is not a skin-colour/tone off switch. Use the final skin/scene controls below for that.",
                  "-1 segue a estrutura local; 0 reduz a estrutura na pele de forma independente."));
            bool mask = config->DlssNrAutoMask.value_or_default();
            if (ImGui::Checkbox(T("Auto skin mask", "Máscara automática de pele"), &mask))
                config->DlssNrAutoMask = mask;
            HelpMarker(
                T("NVIDIA's internal automatic mask, not the colour-based preview below."
                  "\nWith Skin structure=-1 it follows general structure, so the difference may be subtle."
                  "\nTry Skin structure=0 versus Local structure=1 to compare. Mask accuracy is model-dependent.",
                  "Máscara semântica automática interna da NVIDIA para pele."));
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx(T("Pass 2", "Passe 2"), ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::TextWrapped(
                "%s",
                T("Unset controls inherit pass 1, except local tone which defaults to 0. Reset restores this behavior. "
                  "Only active passes run.",
                  "Controles não definidos herdam do passe 1 (tom local padrão 0). Apenas passes ativos executam."));
            InheritedProfileCombo(T("Style", "Estilo"), &config->DlssNrPass2Style,
                                  ptBr ? inheritedStylesPt : inheritedStylesEn, 4);
            DeferredSlider(T("Intensity##p2", "Intensidade##p2"), &config->DlssNrPass2Intensity, 0.0f, 2.0f,
                           config->DlssNrIntensity.value_or_default(), "%.2f", true, resetText);
            DeferredSlider(T("Local structure##p2", "Estrutura local##p2"), &config->DlssNrPass2LocalStructure, 0.0f,
                           2.0f, config->DlssNrLocalStructure.value_or_default(), "%.2f", true, resetText);
            DeferredSlider(T("Local tone##p2", "Tom local##p2"), &config->DlssNrPass2LocalTone, 0.0f, 2.0f, 0.0f,
                           "%.2f", true, resetText);
            DeferredSlider(T("Skin structure##p2", "Estrutura de pele##p2"), &config->DlssNrPass2SkinStructure, -1.0f,
                           2.0f, config->DlssNrSkinStructure.value_or_default(), "%.2f", true, resetText);
            bool mask = config->DlssNrPass2AutoMask.has_value() ? config->DlssNrPass2AutoMask.value()
                                                                : config->DlssNrAutoMask.value_or_default();
            if (ImGui::Checkbox(T("Auto skin mask##p2", "Máscara automática de pele##p2"), &mask))
                config->DlssNrPass2AutoMask = mask;
            ImGui::SameLine();
            if (ImGui::SmallButton(T("Reset##mask2", "Restaurar##mask2")))
                config->DlssNrPass2AutoMask = std::optional<bool> {};
            ImGui::TreePop();
        }

        if (ImGui::TreeNodeEx(T("Pass 3", "Passe 3"), ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::TextWrapped(
                "%s",
                T("Unset controls inherit pass 1, except local tone which defaults to 0. Reset restores this behavior. "
                  "Only active passes run.",
                  "Controles não definidos herdam do passe 1 (tom local padrão 0). Apenas passes ativos executam."));
            InheritedProfileCombo(T("Style", "Estilo"), &config->DlssNrPass3Style,
                                  ptBr ? inheritedStylesPt : inheritedStylesEn, 4);
            DeferredSlider(T("Intensity##p3", "Intensidade##p3"), &config->DlssNrPass3Intensity, 0.0f, 2.0f,
                           config->DlssNrIntensity.value_or_default(), "%.2f", true, resetText);
            DeferredSlider(T("Local structure##p3", "Estrutura local##p3"), &config->DlssNrPass3LocalStructure, 0.0f,
                           2.0f, config->DlssNrLocalStructure.value_or_default(), "%.2f", true, resetText);
            DeferredSlider(T("Local tone##p3", "Tom local##p3"), &config->DlssNrPass3LocalTone, 0.0f, 2.0f, 0.0f,
                           "%.2f", true, resetText);
            DeferredSlider(T("Skin structure##p3", "Estrutura de pele##p3"), &config->DlssNrPass3SkinStructure, -1.0f,
                           2.0f, config->DlssNrSkinStructure.value_or_default(), "%.2f", true, resetText);
            bool mask = config->DlssNrPass3AutoMask.has_value() ? config->DlssNrPass3AutoMask.value()
                                                                : config->DlssNrAutoMask.value_or_default();
            if (ImGui::Checkbox(T("Auto skin mask##p3", "Máscara automática de pele##p3"), &mask))
                config->DlssNrPass3AutoMask = mask;
            ImGui::SameLine();
            if (ImGui::SmallButton(T("Reset##mask3", "Restaurar##mask3")))
                config->DlssNrPass3AutoMask = std::optional<bool> {};
            ImGui::TreePop();
        }

        const unsigned int visiblePasses =
            std::clamp(std::max(config->DlssNrPasses.value_or_default(), config->DlssNrRRPasses.value_or_default()), 1u,
                       passLimit);
        for (unsigned int pass = 3; pass < visiblePasses; ++pass)
        {
            auto& settings = config->DlssNrExtraPasses[pass - 3];
            const std::string passTitle = ptBr ? std::format("Passe {}", pass + 1) : std::format("Pass {}", pass + 1);
            if (!ImGui::TreeNode(passTitle.c_str()))
                continue;
            ImGui::TextWrapped("%s", T("Unset controls inherit pass 1; local tone defaults to 0.",
                                       "Controles não definidos herdam do passe 1; tom local padrão 0."));
            InheritedProfileCombo(T("Style", "Estilo"), &settings.style, ptBr ? inheritedStylesPt : inheritedStylesEn,
                                  4);
            DeferredSlider(std::format("{}##p{}", T("Intensity", "Intensidade"), pass + 1).c_str(), &settings.intensity,
                           0.0f, 2.0f, config->DlssNrIntensity.value_or_default(), "%.2f", true, resetText);
            DeferredSlider(std::format("{}##p{}", T("Local structure", "Estrutura local"), pass + 1).c_str(),
                           &settings.structure, 0.0f, 2.0f, config->DlssNrLocalStructure.value_or_default(), "%.2f",
                           true, resetText);
            DeferredSlider(std::format("{}##p{}", T("Local tone", "Tom local"), pass + 1).c_str(), &settings.tone, 0.0f,
                           2.0f, 0.0f, "%.2f", true, resetText);
            DeferredSlider(std::format("{}##p{}", T("Skin structure", "Estrutura de pele"), pass + 1).c_str(),
                           &settings.skin, -1.0f, 2.0f, config->DlssNrSkinStructure.value_or_default(), "%.2f", true,
                           resetText);
            bool mask = settings.autoMask.value_or(config->DlssNrAutoMask.value_or_default());
            if (ImGui::Checkbox(
                    std::format("{}##p{}", T("Auto skin mask", "Máscara automática de pele"), pass + 1).c_str(), &mask))
                settings.autoMask = mask;
            ImGui::SameLine();
            if (ImGui::SmallButton(std::format("{}##mask{}", resetText, pass + 1).c_str()))
                settings.autoMask = std::optional<bool> {};
            ImGui::TreePop();
        }

        if (ImGui::TreeNode(T("Advanced presets & DLSS 5 options", "Predefinições e Ajustes Avançados (DLSS 5)")))
        {
            ImGui::TextWrapped("%s", T("Preset hints and neural parameters for DLSS 5 (310.9.16+). In 310.9.16, "
                                       "Default selects the new fast FP8 model (Preset B).",
                                       "Predefinições e parâmetros neurais do DLSS 5 (310.9.16+). Na versão 310.9.16, "
                                       "o Padrão seleciona o novo modelo otimizado FP8 (Preset B)."));

            static const char* presetsEn[] = { "Default (Auto / 310.9.16 FP8)", "Preset A (Legacy 310.8)",
                                               "Preset B (Fast FP8)", "Preset C (Low Latency)" };
            static const char* presetsPt[] = { "Padrão (Auto / 310.9.16 FP8)", "Preset A (Legado 310.8)",
                                               "Preset B (Rápido FP8)", "Preset C (Baixa Latência)" };
            static const char* inheritedPresetsEn[] = { "Auto (inherit pass 1)", "Default (Auto / 310.9.16 FP8)",
                                                        "Preset A (Legacy 310.8)", "Preset B (Fast FP8)",
                                                        "Preset C (Low Latency)" };
            static const char* inheritedPresetsPt[] = { "Auto (herdar do passe 1)", "Padrão (Auto / 310.9.16 FP8)",
                                                        "Preset A (Legado 310.8)", "Preset B (Rápido FP8)",
                                                        "Preset C (Baixa Latência)" };

            int preset = (int) std::min(config->DlssNrPreset.value_or_default(), 3u);
            if (ImGui::Combo(T("Pass 1 preset hint", "Predefinição do Passe 1"), &preset, ptBr ? presetsPt : presetsEn,
                             4))
                config->DlssNrPreset = (uint32_t) preset;
            InheritedProfileCombo(T("Pass 2 preset hint", "Predefinição do Passe 2"), &config->DlssNrPass2Preset,
                                  ptBr ? inheritedPresetsPt : inheritedPresetsEn, 5);
            InheritedProfileCombo(T("Pass 3 preset hint", "Predefinição do Passe 3"), &config->DlssNrPass3Preset,
                                  ptBr ? inheritedPresetsPt : inheritedPresetsEn, 5);

            // DLSS 5 (310.9.16+) Mask Blend Mode
            static const char* maskModesEn[] = { "Default (Auto)", "Mode 1 (Alternative)", "Mode 2 (Smooth)" };
            static const char* maskModesPt[] = { "Padrão (Automático)", "Modo 1 (Alternativo)", "Modo 2 (Suave)" };
            int maskBlend = (int) std::min(config->DlssNrMaskBlendMode.value_or_default(), 2u);
            if (ImGui::Combo(T("Mask blend mode (DLSS 5)", "Modo de mesclagem da máscara (DLSS 5)"), &maskBlend,
                             ptBr ? maskModesPt : maskModesEn, 3))
            {
                config->DlssNrMaskBlendMode = (uint32_t) maskBlend;
            }
            HelpMarker(T("DLSS 5 310.9.16+ parameter (DLSSNR.MaskBlendMode).\nControls the blending behavior of the "
                         "neural mask kernel with the frame.",
                         "Parâmetro novo da DLL 310.9.16 do DLSS 5 (DLSSNR.MaskBlendMode).\nControla o modo de "
                         "mesclagem do kernel de máscara neural com o quadro."));

            ImGui::TreePop();
        }
        ImGui::TextWrapped("%s", T("Per-pass overrides apply to the DX12 multipass path, including NR after RR. Native "
                                   "Vulkan and the driver-proxy backend remain single-pass.",
                                   "Substituições por passe aplicam-se ao modo multipasse DX12 (incluindo NR após RR). "
                                   "Vulkan nativo e proxy do driver permanecem em passe único."));

        ImGui::SeparatorText(T("Colour", "Cor e Tonalidade"));

        if (ImGui::TreeNode(T("Skin and environment (final edit)", "Pele e Cenário (Composição final)")))
        {
            ImGui::TextWrapped("%s", T("Optional colour-based selection, NOT NVIDIA's automatic skin mask. Warm "
                                       "scenery can be selected and coloured lighting can hide skin. Check the "
                                       "preview. These controls affect the combined result of all passes.",
                                       "Seleção opcional baseada em cor (independente da máscara semântica interna). "
                                       "Ajusta o balanço de pele e cenário no resultado final de todos os passes."));
            bool filter = config->DlssNrSkinProtection.value_or_default();
            if (ImGui::Checkbox(T("Separate skin / environment controls", "Separar controles de pele e cenário"),
                                &filter))
                config->DlssNrSkinProtection = filter;
            ImGui::BeginDisabled(!filter);
            bool tone = config->DlssNrSkinToneEnabled.value_or_default();
            if (ImGui::Checkbox(T("Allow skin tone / colour changes", "Permitir alterações de tom/cor de pele"), &tone))
                config->DlssNrSkinToneEnabled = tone;
            HelpMarker(T("Off preserves colour in selected pixels; lighting/detail can still change."
                         "\nAlso set Skin detail / lighting to 0 to suppress both.",
                         "Desmarcado preserva a tonalidade original nos pixels de pele."));
            const auto slider = [](const char* label, auto& option)
            {
                float v = option.value_or_default();
                if (ImGui::SliderFloat(label, &v, 0.0f, 1.0f, "%.2f"))
                    option = v;
            };
            slider(T("Skin detail / lighting", "Detalhe / iluminação de pele"), config->DlssNrSkinDetail);
            ImGui::BeginDisabled(!tone);
            slider(T("Skin colour", "Cor de pele"), config->DlssNrSkinColour);
            ImGui::EndDisabled();
            slider(T("Environment detail / lighting", "Detalhe / iluminação do cenário"),
                   config->DlssNrEnvironmentDetail);
            slider(T("Environment colour", "Cor do cenário"), config->DlssNrEnvironmentColour);
            bool preview = config->DlssNrShowSkinMask.value_or_default();
            if (ImGui::Checkbox(T("Preview colour-based mask", "Pré-visualizar máscara baseada em cor"), &preview))
                config->DlssNrShowSkinMask = preview;
            ImGui::EndDisabled();
            ImGui::TreePop();
        }

        ImGui::TextDisabled("%s", T("The model was trained on finished, sRGB-encoded frames. The upscaler's\n"
                                    "output is not one: it is linear and open-ended. These decide how it is\n"
                                    "mapped into something the model recognises. A frame the game reports as\n"
                                    "already tone-mapped is passed over untouched and none of this applies.",
                                    "O modelo foi treinado em quadros sRGB finalizados. A saída do upscaler\n"
                                    "é HDR linear. Estas opções definem como a imagem é mapeada para o modelo.\n"
                                    "Quadros já mapeados por tom são repassados sem alteração."));

        {
            {
                const auto ex = DlssNr::GameExposureStatus();
                const bool vk = DlssNr::IsRunningVk();
                const bool haveExposure = vk ? DlssNr::ExposureOfferedVk() : ex.everOffered;

                const float anchorNow = DlssNr::ExposureScan::BestValue();
                const bool haveAnchor = !DlssNr::ExposureScan::Anchors().empty();

                static const char* sourceNamesEn[] = { "Paper white only", "The game's own exposure",
                                                       "A buffer the scan found" };
                static const char* sourceNamesPt[] = { "Apenas branco do papel (manual)", "Exposição nativa do jogo",
                                                       "Buffer detectado pela varredura" };

                int source = (int) config->DlssNrWhitePointSource.value_or_default();

                if (source < 0 || source > 2)
                    source = 0;

                if (ImGui::Combo(T("White point from", "Origem do ponto branco"), &source,
                                 ptBr ? sourceNamesPt : sourceNamesEn, 3))
                {
                    config->DlssNrWhitePointSource = (uint32_t) source;
                }

                HelpMarker(T("Where the number that divides the frame comes from."
                             "\n\nPaper white only -- the slider below and nothing else. Right for a"
                             "\ngame whose exposure never moves, wrong the moment it does: one"
                             "\nconstant cannot serve a cave and a field."
                             "\n\nThe game's own exposure -- read from the texture the game hands"
                             "\nthe upscaler. The best source there is, because it is decided"
                             "\nupstream and nothing this pass does can move it. Not every game"
                             "\nsupplies one."
                             "\n\nA buffer the scan found -- for games that compute an exposure and"
                             "\nnever pass it on. A GUESS: candidates are matched by shape, and in"
                             "\nGTA V the best one tracks the real exposure but at its own scale,"
                             "\nwhich the anchor's ratio cancels. Needs anchoring once, and checking"
                             "\nafterwards.",
                             "Origem do valor que normaliza o quadro antes da inferência do modelo."
                             "\nA exposição nativa do jogo é a melhor opção quando suportada."));

                if (source == 1)
                {
                    if (!vk && ex.seenFrames == 0)
                        ImGui::TextDisabled("%s", T("Waiting for a frame...", "Aguardando um quadro..."));
                    else if (!haveExposure)
                        ImGui::TextColored(
                            ImVec4(0.9f, 0.6f, 0.25f, 1.0f), "%s",
                            T("This game supplies no exposure -- paper white is in use. Try the scan instead.",
                              "Este jogo não fornece exposição -- usando branco do papel manual. Experimente a "
                              "varredura."));
                    else if (vk)
                        ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f), "%s",
                                           T("This game supplies an exposure and it is being read.",
                                             "Este jogo fornece textura de exposição ativa e ela está sendo lida."));
                    else if (ex.exposure > 1e-6f)
                    {
                        const float trim = std::clamp(config->DlssNrWhitePointTrim.value_or_default(), 0.25f, 4.0f);
                        ImGui::TextColored(
                            ImVec4(0.45f, 0.8f, 0.45f, 1.0f),
                            T("Game exposure %.4f  ->  white point %.2f%s",
                              "Exposição do jogo %.4f  ->  ponto branco %.2f%s"),
                            ex.exposure, ex.preExposure / ex.exposure * trim,
                            ex.offeredNow ? "" : T("  (held: absent this frame)", "  (mantido: ausente neste quadro)"));
                    }
                    else
                        ImGui::TextDisabled("%s", T("Reading the exposure...", "Lendo exposição..."));
                }
                else if (source == 2)
                {
                    if (anchorNow <= 0.0f)
                    {
                        const unsigned int watching = (unsigned int) DlssNr::ExposureScan::Report().size();

                        if (watching == 0)
                            ImGui::TextColored(ImVec4(0.9f, 0.6f, 0.25f, 1.0f), "%s",
                                               T("Nothing in this game is shaped like an exposure.",
                                                 "Nenhum buffer neste jogo possui formato compatível com exposição."));
                        else
                            ImGui::TextColored(
                                ImVec4(0.9f, 0.6f, 0.25f, 1.0f),
                                T("Watching %u, none moving yet -- go between light and shade.",
                                  "Monitorando %u buffers, nenhum se moveu ainda -- alterne entre luz e sombra."),
                                watching);
                    }
                    else if (!haveAnchor)
                        ImGui::TextColored(
                            ImVec4(0.9f, 0.6f, 0.25f, 1.0f), "%s",
                            T("Found one. Set paper white below until the picture looks right, then press Anchor here.",
                              "Buffer detectado. Ajuste o branco do papel abaixo até a imagem ficar correta e clique "
                              "em Ancorar ponto aqui."));
                }
                else if (haveExposure)
                {
                    ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f), "%s",
                                       T("This game supplies an exposure -- the option above would use it.",
                                         "Este jogo fornece exposição nativa -- a opção acima a utilizaria."));
                }
            }

            const int wpSource = (int) config->DlssNrWhitePointSource.value_or_default();

            static int selectedAnchor = -1;
            auto anchors = DlssNr::ExposureScan::Anchors();
            if (selectedAnchor >= (int) anchors.size())
                selectedAnchor = -1;

            if (wpSource == 2)
            {
                const bool editingRow = selectedAnchor >= 0 && selectedAnchor < (int) anchors.size();

                if (!anchors.empty())
                {
                    const float liveScan = DlssNr::ExposureScan::BestValue();

                    if (liveScan > 0.0f)
                    {
                        const float w = DlssNr::ExposureScan::AnchoredWhitePoint(
                            liveScan, config->DlssNrScanInverted.value_or_default(),
                            config->DlssNrScanTrim.value_or_default());

                        ImGui::TextColored(ImVec4(0.45f, 0.8f, 0.45f, 1.0f),
                                           T("Scan %.5f  ->  white point %.2f   (%u point%s)",
                                             "Varredura %.5f  ->  ponto branco %.2f   (%u ponto%s)"),
                                           liveScan, w, (unsigned) anchors.size(),
                                           anchors.size() == 1 ? "" : (ptBr ? "s" : "s"));
                    }
                }

                const bool showPaperWhite = anchors.empty() || editingRow;

                if (showPaperWhite)
                {
                    float pw =
                        editingRow ? anchors[selectedAnchor].white : config->DlssNrWhitePointScale.value_or_default();

                    char lbl[64];
                    if (editingRow)
                        snprintf(lbl, sizeof(lbl), "%s (point %d)##pwed", T("Paper white", "Branco do papel"),
                                 selectedAnchor + 1);
                    else
                        snprintf(lbl, sizeof(lbl), "%s##pw", T("Paper white", "Branco do papel"));

                    if (ImGui::SliderFloat(lbl, &pw, 0.25f, 2000.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
                    {
                        if (editingRow)
                        {
                            DlssNr::ExposureScan::AnchorSetWhite(selectedAnchor, pw);
                            config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                        }
                        else
                            config->DlssNrWhitePointScale = pw;
                    }

                    HelpMarker(T("The white point for the selected calibration point, or -- with no row"
                                 "\nselected -- the value the next Anchor press captures."
                                 "\n\nSet it until the picture looks right here, then Anchor. Move to very"
                                 "\ndifferent light and do it again: two points fix the buffer's real"
                                 "\nrelationship and the white point holds between them.",
                                 "Ponto branco para o ponto de calibração selecionado."));
                }

                if (!anchors.empty())
                {
                    float trim = config->DlssNrScanTrim.value_or_default();

                    if (ImGui::SliderFloat(T("Trim (x the scan)", "Ajuste fino (x varredura)"), &trim, 0.25f, 4.0f,
                                           "%.2fx", ImGuiSliderFlags_Logarithmic))
                        config->DlssNrScanTrim = std::clamp(trim, 0.25f, 4.0f);

                    ImGui::SameLine();

                    if (ImGui::SmallButton(T("Reset##scantrim", "Restaurar##scantrim")))
                        config->DlssNrScanTrim = 1.0f;

                    HelpMarker(T("A multiplier on the scan's white point, and the control you adjust between"
                                 "\nanchor points: dial it until the picture looks right in the current"
                                 "\nlight, then press Anchor here -- it captures the trimmed value as a new"
                                 "\npoint and resets the trim to 1.",
                                 "Multiplicador no ponto branco da varredura para ajuste fino entre âncoras."));
                }
            }
            else if (wpSource == 1)
            {
                float trim = config->DlssNrWhitePointTrim.value_or_default();

                if (ImGui::SliderFloat(T("Trim (x the game's exposure)", "Ajuste fino (x exposição do jogo)"), &trim,
                                       0.25f, 4.0f, "%.2fx", ImGuiSliderFlags_Logarithmic))
                {
                    config->DlssNrWhitePointTrim = std::clamp(trim, 0.25f, 4.0f);
                }

                ImGui::SameLine();

                if (ImGui::SmallButton(T("Reset##wptrim", "Restaurar##wptrim")))
                    config->DlssNrWhitePointTrim = 1.0f;

                HelpMarker(T("A multiplier on the exposure the game supplied. 1.00x takes its number"
                             "\nexactly, and that is the right answer here."
                             "\n\nThis is not a fudge factor. If a game needs the trim far from 1 to look"
                             "\nright, that is evidence the exposure being read is wrong for that game."
                             "\nSomewhere around 0.8 to 1.25 is honest tuning.",
                             "Multiplicador sobre a exposição do jogo (1.00x usa o valor exato)."));
            }
            else
            {
                float wpScale = config->DlssNrWhitePointScale.value_or_default();

                if (ImGui::SliderFloat(T("Paper white", "Branco do papel (Paper white)"), &wpScale, 0.25f, 2000.0f,
                                       "%.2fx", ImGuiSliderFlags_Logarithmic))
                    config->DlssNrWhitePointScale = wpScale;

                HelpMarker(T("What the frame is divided by before the model sees it. There is no other white"
                             "\npoint; this is the whole of it."
                             "\n\nThe model was trained on finished frames where white sits at 1. The"
                             "\nupscaler's output is linear and open-ended, so something has to say where"
                             "\nwhite is.",
                             "Divisor de normalização do quadro antes da inferência da rede neural."));
            }

            float maxRatio = config->DlssNrMaxRatio.value_or_default();
            if (ImGui::SliderFloat(T("Highlight guard", "Proteção de realces (Highlight guard)"), &maxRatio, 1.0f,
                                   unlockPasses ? (float) MaxPassCount : 8.0f, "%.1fx"))
                config->DlssNrMaxRatio = maxRatio;

            ImGui::SameLine();
            if (ImGui::SmallButton(T("Reset##guard", "Restaurar##guard")))
                config->DlssNrMaxRatio = 2.0f;

            HelpMarker(T("The most the pass may move any pixel, as a multiple of what it already was, in"
                         "\nboth directions -- a pixel may not be brightened past this nor darkened past"
                         "\nits reciprocal. 2x leaves detail intact while stopping a strip light turning"
                         "\ninto a string of coloured cells.",
                         "Limite de alteração de luminância por pixel para evitar estouro de luzes fortes."));

            {
                bool meter = config->DlssNrScanMeter.value_or_default();

                if (config->DlssNrWhitePointSource.value_or_default() == 2 &&
                    ImGui::Checkbox(T("Show the light meter on screen", "Exibir fotômetro na tela"), &meter))
                    config->DlssNrScanMeter = meter;

                HelpMarker(T("A lamp in the corner: red for dark, green for full light, and the"
                             "\nshades between, with the reading beside it."
                             "\n\nIt is how you see at a glance that the scan is TRACKING rather"
                             "\nthan merely running.",
                             "Indicador visual no canto da tela mostrando a medição da iluminação."));

                if (DlssNr::ExposureScan::Scanning())
                {
                    int which = 0;
                    float low = 0.0f, high = 0.0f;
                    const float live = DlssNr::ExposureScan::BestValue(&which, &low, &high);

                    const bool isSource = config->DlssNrWhitePointSource.value_or_default() == 2;

                    ImGui::BeginDisabled(live <= 0.0f || !isSource);

                    if (ImGui::Button(T("Anchor here", "Ancorar ponto aqui")))
                    {
                        const float captureWhite =
                            anchors.empty() ? std::max(0.01f, config->DlssNrWhitePointScale.value_or_default())
                                            : std::max(0.01f, DlssNr::ExposureScan::AnchoredWhitePoint(
                                                                  live, config->DlssNrScanInverted.value_or_default(),
                                                                  config->DlssNrScanTrim.value_or_default()));

                        if (DlssNr::ExposureScan::AnchorAdd(live, captureWhite))
                        {
                            config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                            config->DlssNrScanTrim = 1.0f;
                            selectedAnchor = -1;
                        }
                    }

                    ImGui::EndDisabled();

                    HelpMarker(T("Make the picture look right, then press this -- it captures the current look"
                                 "\nas a point. For the first point use the Paper white slider above; for"
                                 "\nevery point after, move to different light and use the Trim.",
                                 "Captura o ponto de calibração atual para interpolação da iluminação."));

                    if (!isSource)
                        ImGui::TextDisabled(
                            "%s",
                            T("(the scan is only watching -- the white point above comes from somewhere else)",
                              "(a varredura está apenas monitorando -- o ponto branco acima vem de outra fonte)"));

                    if (!anchors.empty())
                    {
                        int active = 0;
                        float bestDist = 1e30f;
                        const float liveLog = std::log(std::max(live, 1e-6f));

                        for (size_t i = 0; i < anchors.size(); ++i)
                        {
                            const float d = std::fabs(std::log(std::max(anchors[i].scan, 1e-6f)) - liveLog);
                            if (d < bestDist)
                            {
                                bestDist = d;
                                active = (int) i;
                            }
                        }

                        for (size_t i = 0; i < anchors.size(); ++i)
                        {
                            ImGui::PushID((int) i);

                            if (ImGui::SmallButton("x"))
                            {
                                DlssNr::ExposureScan::AnchorRemove((int) i);
                                config->DlssNrScanAnchors = DlssNr::ExposureScan::SerializeAnchors();
                                if (selectedAnchor == (int) i)
                                    selectedAnchor = -1;
                                else if (selectedAnchor > (int) i)
                                    --selectedAnchor;
                                ImGui::PopID();
                                continue;
                            }

                            ImGui::SameLine();

                            const bool sel = (int) i == selectedAnchor;
                            char row[96];
                            snprintf(row, sizeof(row), "%s %s %.4f  ->  %s %.2f%s",
                                     ((int) i == active && isSource) ? ">" : "  ", T("scan", "varredura"),
                                     anchors[i].scan, T("white", "branco"), anchors[i].white,
                                     sel ? T("   [editing]", "   [editando]") : "");

                            if (ImGui::Selectable(row, sel))
                                selectedAnchor = sel ? -1 : (int) i;

                            ImGui::PopID();
                        }

                        ImGui::TextDisabled("%s",
                                            T("Click a row to edit it with the slider above; click it again to control "
                                              "the live point. > is the point in use now.",
                                              "Clique em uma linha para editá-la com o controle deslizante; clique "
                                              "novamente para voltar ao ponto dinâmico. > indica o ponto ativo."));
                    }

                    if (anchors.size() == 1)
                    {
                        bool inverted = config->DlssNrScanInverted.value_or_default();
                        if (ImGui::Checkbox(
                                T("The number runs the other way", "Inverter direção do valor da varredura"),
                                &inverted))
                            config->DlssNrScanInverted = inverted;

                        HelpMarker(
                            T("Flip this if the picture gets worse in the direction it should be getting better.",
                              "Inverta caso a imagem escureça onde deveria clarear."));
                    }

                    if (ImGui::TreeNode(T("Advanced", "Avançado (Detalhes da varredura)")))
                    {
                        const auto found = DlssNr::ExposureScan::Report();
                        const char* why = DlssNr::ExposureScan::Status();

                        if (found.empty())
                        {
                            ImGui::TextDisabled("%s",
                                                why != nullptr && why[0] != 0
                                                    ? why
                                                    : T("nothing matched yet.", "nenhum buffer compatível ainda."));
                        }
                        else
                        {
                            for (size_t i = 0; i < found.size(); ++i)
                            {
                                const auto& c = found[i];

                                if (c.reads == 0)
                                {
                                    ImGui::TextDisabled("%zu. %s -- %s", i + 1, c.shape.c_str(),
                                                        T("not read yet", "não lido ainda"));
                                    continue;
                                }

                                ImGui::TextColored(
                                    c.moves ? ImVec4(0.45f, 0.8f, 0.45f, 1.0f) : ImVec4(0.6f, 0.6f, 0.6f, 1.0f),
                                    "%zu. %s = %.5f  (%s %.5f..%.5f) %s", i + 1, c.shape.c_str(), c.latest,
                                    T("seen", "faixa"), c.lowest, c.highest,
                                    c.moves ? T("MOVES", "MOVE-SE") : T("flat so far", "estático até agora"));
                            }

                            ImGui::TextDisabled(
                                "%s", T("Walk from shade into daylight. A real exposure moves.",
                                        "Alterne entre sombra e luz do dia para verificar se a exposição se move."));
                        }

                        ImGui::TreePop();
                    }
                }
            }
        }

        ImGui::SeparatorText(T("Compare", "Comparação e Análise"));

        bool held = config->DlssNrHoldFrame.value_or_default();
        if (ImGui::Checkbox(T("Hold frame", "Congelar quadro (Hold frame)"), &held))
            config->DlssNrHoldFrame = held;

        HelpMarker(T("Freezes the frame the model works on. While held, change paper white, the"
                     "\nstrengths, the reversible mode, the model preset -- anything below the"
                     "\nupscaler -- and only that setting moves; the scene does not."
                     "\n\nHide the menu and it stays held. Untoggle to resume.",
                     "Congela o quadro de entrada do modelo. Permite alternar e ajustar configurações"
                     "\nno mesmo quadro estático para comparação perfeita A/B."));

        static const char* compareNamesEn[] = { "Off", "Side by side", "Wipe" };
        static const char* compareNamesPt[] = { "Desativado", "Lado a lado", "Divisão deslizante (Wipe)" };
        int compare = (int) config->DlssNrCompare.value_or_default();
        if (ImGui::Combo(T("Compare", "Modo de comparação"), &compare, ptBr ? compareNamesPt : compareNamesEn, 3))
            config->DlssNrCompare = (uint32_t) compare;

        HelpMarker(
            T("Shows the pass against itself, so the two can be seen at once rather than toggled and remembered."
              "\n\nSide by side puts the whole frame in each half, untouched on the left and edited on the right."
              "\n\nWipe cuts a single frame at the split and resamples nothing, so the picture can be played normally.",
              "Exibe o quadro com e sem Neural Rendering simultaneamente em tela."));

        if (compare != 0)
        {
            bool swap = config->DlssNrCompareSwap.value_or_default();
            if (ImGui::Checkbox(T("Swap sides", "Inverter lados"), &swap))
                config->DlssNrCompareSwap = swap;

            bool tags = config->DlssNrCompareTags.value_or_default();
            if (ImGui::Checkbox(T("Label the sides", "Exibir rótulos nos lados"), &tags))
                config->DlssNrCompareTags = tags;

            HelpMarker(T("Writes which side is which onto the frame itself, so a screenshot still says so.",
                         "Desenha textos identificando cada lado da imagem para capturas de tela."));

            if (tags)
            {
                float tagScale = config->DlssNrTagScale.value_or_default();
                if (ImGui::SliderFloat(T("Label size", "Tamanho do rótulo"), &tagScale, 0.5f, 5.0f, "%.1fx"))
                    config->DlssNrTagScale = std::clamp(tagScale, 0.5f, 5.0f);
            }
        }

        if (compare == 1)
        {
            float zoom = config->DlssNrCompareZoom.value_or_default();
            if (ImGui::SliderFloat(T("Zoom", "Zoom"), &zoom, 1.0f, 2.0f, "%.2f"))
                config->DlssNrCompareZoom = std::clamp(zoom, 1.0f, 2.0f);
        }

        if (compare == 2)
        {
            float split = config->DlssNrCompareSplit.value_or_default();
            if (ImGui::SliderFloat(T("Split", "Posição da divisão"), &split, 0.0f, 1.0f, "%.2f"))
                config->DlssNrCompareSplit = std::clamp(split, 0.0f, 1.0f);
        }

        static const char* debugNamesEn[] = { "Off", "Proxy (what the model sees)", "Model output (raw)",
                                              "Difference (amplified)" };
        static const char* debugNamesPt[] = { "Desativado", "Proxy (o que o modelo vê)", "Saída do modelo (bruta)",
                                              "Diferença (amplificada 20x)" };
        int debugView = (int) config->DlssNrDebugView.value_or_default();
        if (ImGui::Combo(T("Debug view", "Visualização de depuração"), &debugView, ptBr ? debugNamesPt : debugNamesEn,
                         4))
            config->DlssNrDebugView = (uint32_t) debugView;

        HelpMarker(
            T("Proxy is the picture handed to the model -- if that looks wrong, the white point is wrong."
              "\n\nDifference shows what the model actually changed, amplified twenty times and centred on grey.",
              "Proxy exibe a imagem entregue ao modelo. Diferença exibe o que a rede alterou amplificado 20x."));

        ImGui::PopItemWidth();
    }
}

} // namespace DlssNr
