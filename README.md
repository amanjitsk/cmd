# Conformal Mirror Descent

Code for [Conformal Mirror
Descent](https://link.springer.com/article/10.1007/s41884-022-00089-3)
(Information Geometry 2022).

Install the requirements into a local virtual environment using:

    pip install -r requirements.txt

To run the Student t-distribution example (with k = 3), run the
following command:

    python student_online.py --k 3

Similarly, the Dirichlet perturbation example can be run with
`python dirichlet_perturbation_model.py`. The Dirichlet transport
examples in Section 5 can be run with

    python dirichlet_transport --name NAME

where `NAME` can be one of `center`, `dirichlet-1`, `dirichlet-2` or
`random`.

If you find this useful, please consider citing:

    @Article{kainth2022cmd,
      abstract={The logarithmic divergence is an extension of the Bregman divergence motivated by optimal
    transport and a generalized convex duality, and satisfies many remarkable properties. Using the geometry
    induced by the logarithmic divergence, we introduce a generalization of continuous time mirror descent
    that we term the conformal mirror descent. We derive its dynamics under a generalized mirror map, and
    show that it is a time change of a corresponding Hessian gradient flow. We also prove convergence results
    in continuous time. We apply the conformal mirror descent to online estimation of a generalized
    exponential family, and construct a family of gradient flows on the unit simplex via the Dirichlet
    optimal transport problem.},
      author={Kainth, Amanjit Singh and Wong, Ting-Kam Leonard and Rudzicz, Frank},
      doi={10.1007/s41884-022-00089-3},
      journal={Information Geometry},
      language={en},
      month={12},
      publisher={Springer Science and Business Media LLC},
      title={Conformal mirror descent with logarithmic divergences},
      url={http://dx.doi.org/10.1007/s41884-022-00089-3},
      year={2022}
    }
