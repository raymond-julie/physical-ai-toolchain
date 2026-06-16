package com.trainmybot.camerastream.impl;

import java.util.Locale;

import com.ur.urcap.api.contribution.ViewAPIProvider;
import com.ur.urcap.api.contribution.installation.ContributionConfiguration;
import com.ur.urcap.api.contribution.installation.CreationContext;
import com.ur.urcap.api.contribution.installation.InstallationAPIProvider;
import com.ur.urcap.api.contribution.installation.swing.SwingInstallationNodeService;
import com.ur.urcap.api.domain.data.DataModel;

/**
 * Service definition for the Camera Stream installation node. PolyScope calls
 * into this to create the single installation-scoped contribution and its
 * Swing view.
 */
public class CameraStreamInstallationNodeService
        implements SwingInstallationNodeService<
                CameraStreamInstallationNodeContribution,
                CameraStreamInstallationNodeView> {

    @Override
    public void configureContribution(ContributionConfiguration configuration) {
        // No special configuration required.
    }

    @Override
    public String getTitle(Locale locale) {
        return "Camera Stream";
    }

    @Override
    public CameraStreamInstallationNodeView createView(ViewAPIProvider apiProvider) {
        return new CameraStreamInstallationNodeView();
    }

    @Override
    public CameraStreamInstallationNodeContribution createInstallationNode(
            InstallationAPIProvider apiProvider,
            CameraStreamInstallationNodeView view,
            DataModel model,
            CreationContext context) {
        return new CameraStreamInstallationNodeContribution(apiProvider, view, model);
    }
}
